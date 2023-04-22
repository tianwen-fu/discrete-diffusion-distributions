from typing import Sequence

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp


class DynamicsModel(nn.Module):
    num_classes: int
    cls_embed_dim: int
    time_embed_dim: int
    x_features_in: Sequence[int]  # MLPs before adding time embedding
    x_features_out: Sequence[int]  # MLPs after adding time embedding
    max_time: int

    @nn.compact
    def __call__(self, x, t):
        # shapes
        batch_size = x.shape[0]
        chex.assert_shape(t, (batch_size,))
        chex.assert_type(t, jnp.int32)
        data_shape = x.shape

        # compute the embedding of x
        # x = jax.nn.one_hot(x, self.num_classes)
        x = nn.Embed(
            num_embeddings=self.num_classes, features=self.cls_embed_dim, name="x_embed"
        )(x)
        chex.assert_shape(x, (*data_shape, self.cls_embed_dim))
        for i, feat in enumerate(self.x_features_in):
            x = nn.Dense(feat, name=f"x_in_{i}")(x)
            x = nn.relu(x)

        # compute timestep embedding, copied from the D3PM code
        t = t.astype(jnp.float32) * 1000 / self.max_time
        half_dim = self.time_embed_dim // 2
        time_emb = jnp.log(10000.0) / (half_dim - 1)
        time_emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -time_emb)
        time_emb = t[:, None] * time_emb[None, :]
        time_emb = jnp.concatenate([jnp.sin(time_emb), jnp.cos(time_emb)], axis=1)
        if self.time_embed_dim % 2 == 1:
            time_emb = jax.lax.pad(time_emb, jnp.float32(0), ((0, 0, 0), (0, 1, 0)))
        chex.assert_shape(time_emb, (batch_size, self.time_embed_dim))
        t = time_emb
        t = nn.Dense(self.time_embed_dim * 4, name="t_mlp1")(t)
        t = nn.Dense(x.shape[-1], name="t_mlp2")(nn.swish(t))

        x = x + t
        for i, feat in enumerate(self.x_features_out):
            x = nn.Dense(feat, name=f"x_out_{i}")(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes, name=f"classifier")(x)
        chex.assert_shape(x, (*data_shape, self.num_classes))
        return x
