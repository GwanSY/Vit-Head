"""The upstream Flax ViT, with its intended BumbleBee GELU binding."""
from contextlib import contextmanager


def load(model_dir, image_path):
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict, unflatten_dict
    from PIL import Image
    from safetensors.numpy import load_file
    from transformers import AutoImageProcessor, FlaxViTForImageClassification, ViTConfig
    from transformers.modeling_flax_pytorch_utils import rename_key_and_reshape_tensor

    config = ViTConfig.from_pretrained(model_dir, local_files_only=True)
    assert (config.num_hidden_layers, config.hidden_size, config.num_attention_heads,
            config.image_size, config.patch_size, config.num_labels) == (12, 768, 12, 224, 16, 1000)
    model = FlaxViTForImageClassification(config, dtype=jnp.float32)
    expected = flatten_dict(model.params)
    converted = {}
    for name, value in load_file(str(model_dir / "model.safetensors")).items():
        key, value = rename_key_and_reshape_tensor(
            tuple(name.split(".")), value, expected, model.base_model_prefix)
        if key not in expected or value.shape != expected[key].shape or key in converted:
            raise ValueError(f"Checkpoint mapping mismatch: {name}")
        converted[key] = jnp.asarray(value, dtype=jnp.float32)
    if converted.keys() != expected.keys():
        raise ValueError("Checkpoint has missing tensors")
    model.params = unflatten_dict(converted)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    with Image.open(image_path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="np")["pixel_values"]
    return model, np.asarray(inputs, dtype=np.float32)


@contextmanager
def secure_gelu():
    import flax.linen as fnn
    import jax.nn as jnn
    from spu.intrinsic import f_gelu
    from transformers.models.vit.modeling_flax_vit import ACT2FN

    # Transformers stores a partial in ACT2FN: patching jax.nn alone misses it.
    saved = jnn.gelu, fnn.gelu, ACT2FN["gelu"]
    jnn.gelu = fnn.gelu = ACT2FN["gelu"] = f_gelu
    try:
        yield
    finally:
        jnn.gelu, fnn.gelu, ACT2FN["gelu"] = saved


def inference(model):
    def vit(params, pixels):
        with secure_gelu():
            return model(pixels, params=params, train=False).logits
    return vit
