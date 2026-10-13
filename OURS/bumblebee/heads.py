"""Public, fixed S/R/L head plans; combine shared matrix stages by default."""
from contextlib import contextmanager


def groups(types):
    return [(kind, [i for i, t in enumerate(types) if t == kind])
            for kind in "SRL" if kind in types]


def mapping(z, kind, alpha=None):
    import jax.numpy as jnp
    import jax.nn as nn
    if kind == "S":
        return nn.softmax(z, axis=-1)
    if kind == "R":
        r = nn.relu(z)
        denominator = r.sum(axis=-1, keepdims=True)
        # Avoid the upstream 16-bit EqualAP for the nonnegative denominator.
        return r / jnp.where(denominator > 0, denominator, 1)
    if kind == "L":
        return z * jnp.asarray(alpha)[None, :, None, None]
    raise ValueError(f"Unknown head type: {kind}")


def attention(q, k, v, types, alpha, layout="joint"):
    import jax.numpy as jnp

    scores = (q / 8) @ jnp.swapaxes(k, -1, -2)
    weights, outputs, order = [], [], []
    for kind, indices in groups(types):
        z = scores[:, indices]
        w = mapping(z, kind, [alpha[i] for i in indices])
        order.extend(indices)
        if layout == "split":
            outputs.append(w @ v[:, indices])
        else:
            weights.append(w)
    inverse = [order.index(i) for i in range(len(types))]
    if layout == "split":
        return jnp.concatenate(outputs, axis=1)[:, inverse]
    w = jnp.concatenate(weights, axis=1)[:, inverse]
    return w @ v


@contextmanager
def bind(plan, layout="joint"):
    from transformers.models.vit.modeling_flax_vit import FlaxViTSelfAttention
    import jax.numpy as jnp

    original = FlaxViTSelfAttention.__call__

    def forward(self, hidden_states, deterministic=True, output_attentions=False):
        layer = int(next(p for p in self.scope.path if str(p).isdigit()))
        types = plan["heads"][layer]
        if set(types) == {"S"}:
            return original(self, hidden_states, deterministic, output_attentions)
        if not deterministic or output_attentions:
            raise ValueError("The secure head path is inference-only")
        shape = hidden_states.shape[:2] + (12, 64)
        q, k, v = [jnp.transpose(project(hidden_states).reshape(shape), (0, 2, 1, 3))
                   for project in (self.query, self.key, self.value)]
        out = attention(q, k, v, types, plan["alpha"][layer], layout)
        return (jnp.transpose(out, (0, 2, 1, 3)).reshape(hidden_states.shape),)

    FlaxViTSelfAttention.__call__ = forward
    try:
        yield
    finally:
        FlaxViTSelfAttention.__call__ = original
