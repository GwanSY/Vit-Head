"""Graph-only scheduling changes; the existing secure primitives are unchanged.

``organized`` coalesces public head selection into slices and skips identity
restoration. ``batched`` additionally normalizes S/R heads together. Every
logical head retains its original type, reduction axis and fixed-point path.
"""
from contextlib import contextmanager
import re


def lower_reciprocal_only(code, variant):
    """Lower explicit 1/d to the same PPHLO reciprocal used by baseline.

    The upstream broadcast-only pass leaves jnp.reciprocal as general DivOp.
    General divide would use a different fixed-point implementation. Restrict
    this local lowering to a verified public constant one with identical shape
    and dtype; execute the unchanged native ReciprocalOp instead.
    """
    if variant != "reciprocal_only": return code
    text = code.decode()
    ones = set(re.findall(r'^    (%[\w#]+) = "pphlo.constant"\(\).*value = dense<1\.000000e\+00>',
                          text,re.M))
    count = 0
    lines = []
    for line in text.splitlines():
        match = re.match(r'^(    %[\w#]+ = )"pphlo.divide"\((%[\w#]+), (%[\w#]+)\) : '
                         r'\((tensor<.+>), (tensor<.+>)\) -> (tensor<.+>)$',line)
        if match and match[2] in ones:
            if '!pphlo.pub<f32>' not in match[4] or match[5] != match[6]:
                raise ValueError("Reciprocal lowering requires verified matching f32 types")
            line = f'{match[1]}"pphlo.reciprocal"({match[3]}) : ({match[5]}) -> {match[6]}'
            count += 1
        lines.append(line)
    if count == 0: raise ValueError("Explicit reciprocal lowering found no verified constant-one division")
    return ("\n".join(lines)+"\n").encode()


def select_public(x, indices):
    """Select fixed public heads, using one slice per contiguous run."""
    import jax.numpy as jnp
    indices = list(indices)
    if indices == list(range(x.shape[1])):
        return x
    if not indices:
        raise ValueError("Empty head selection")
    runs = []
    left = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append(x[:, left:previous + 1])
            left = index
        previous = index
    runs.append(x[:, left:previous + 1])
    return runs[0] if len(runs) == 1 else jnp.concatenate(runs, axis=1)


def organized(q, k, v, types, alpha, layout="joint"):
    import jax.numpy as jnp
    from heads import groups, mapping
    if layout != "joint":
        raise ValueError("This experiment fixes the joint QK/AV layout")
    scores = (q / 8) @ jnp.swapaxes(k, -1, -2)
    order, weights = [], []
    for kind, indices in groups(types):
        weights.append(mapping(select_public(scores, indices), kind,
                               [alpha[i] for i in indices]))
        order.extend(indices)
    joined = weights[0] if len(weights) == 1 else jnp.concatenate(weights, axis=1)
    return select_public(joined, [order.index(i) for i in range(len(types))]) @ v


def batched(q, k, v, types, alpha, layout="joint", normalize_together=True):
    """One existing reciprocal and normalized multiply for all S/R heads.

    JAX Softmax's max/exp/sum sequence is retained, including initial=-inf.
    The R zero-denominator guard is retained. L is kept outside normalization.
    No positive-only reciprocal, precision change or protocol shortcut is used.
    """
    import jax
    import jax.numpy as jnp
    import jax.nn as nn
    from heads import groups, mapping
    if layout != "joint":
        raise ValueError("This experiment fixes the joint QK/AV layout")
    scores = (q / 8) @ jnp.swapaxes(k, -1, -2)
    numerators, denominators, normalized_order = [], [], []
    linear, linear_order = [], []
    for kind, indices in groups(types):
        z = select_public(scores, indices)
        if kind == "L":
            linear.append(mapping(z, kind, [alpha[i] for i in indices]))
            linear_order.extend(indices)
            continue
        if kind == "S":
            maximum = jnp.max(z, axis=-1, initial=-jnp.inf, keepdims=True)
            numerator = jnp.exp(z - jax.lax.stop_gradient(maximum))
            denominator = jnp.sum(numerator, axis=-1, keepdims=True)
        elif kind == "R":
            numerator = nn.relu(z)
            total = jnp.sum(numerator, axis=-1, keepdims=True)
            denominator = jnp.where(total > 0, total, 1)
        else:
            raise ValueError(kind)
        numerators.append(numerator)
        denominators.append(denominator)
        normalized_order.extend(indices)
    weights = []
    if numerators:
        denominator = (denominators[0] if len(denominators) == 1 else
                       jnp.concatenate(denominators, axis=1))
        if normalize_together:
            numerator = (numerators[0] if len(numerators) == 1 else
                         jnp.concatenate(numerators, axis=1))
            # The compiler keeps the reciprocal before broadcasting, as baseline.
            weights.append(numerator / denominator)
        else:
            # A separate candidate batches only the small denominator vectors.
            # Retain the original group-sized normalization multiply batches.
            inverse = jnp.reciprocal(denominator)
            offset = 0
            for numerator in numerators:
                size = numerator.shape[1]
                weights.append(numerator * inverse[:, offset:offset+size])
                offset += size
    weights.extend(linear)
    order = normalized_order + linear_order
    joined = weights[0] if len(weights) == 1 else jnp.concatenate(weights, axis=1)
    return select_public(joined, [order.index(i) for i in range(len(types))]) @ v


def reciprocal_only(q, k, v, types, alpha, layout="joint"):
    return batched(q,k,v,types,alpha,layout,normalize_together=False)


@contextmanager
def bind_variant(plan, variant):
    import heads
    saved = heads.attention
    choices = {"baseline": saved, "organized": organized, "batched": batched,
               "reciprocal_only": reciprocal_only}
    if variant not in choices:
        raise ValueError(variant)
    heads.attention = choices[variant]
    try:
        with heads.bind(plan, "joint"):
            yield
    finally:
        heads.attention = saved
