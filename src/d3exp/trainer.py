import os
import pprint
import time
from functools import partial
from logging import Logger
from typing import Tuple

import chex
import flax
import flax.struct
import jax
import jax.numpy as jnp
import jax.random as jrnd
import optax
from flax.metrics import tensorboard

from d3exp.config import Config
from d3exp.diffusion import CategoricalDiffusion
from d3exp.dynamics import DynamicsModel


@flax.struct.dataclass
class TrainState:
    step: int
    optimizer_state: optax.OptState
    params: flax.core.FrozenDict


@jax.jit
def global_norm(pytree):
    return jnp.sqrt(
        jnp.sum(
            jnp.asarray(
                [jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(pytree)]
            )
        )
    )


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.dynamics = DynamicsModel(
            num_classes=config.num_classes,
            cls_embed_dim=config.cls_embed_dim,
            time_embed_dim=config.time_embed_dim,
            x_features_in=config.x_features_in,
            x_features_out=config.x_features_out,
            max_time=config.num_timesteps,
        )
        self.diffusion = CategoricalDiffusion(config)

    def make_init_params(self, rng):
        return self.dynamics.init(
            rng,
            jnp.zeros(
                (self.config.batch_size, *self.config.data_shape), dtype=jnp.int32
            ),
            jnp.arange(self.config.batch_size, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, batch, rng) -> Tuple[jnp.ndarray, dict]:
        chex.assert_type(batch, jnp.int32)

        model_fn = partial(self.dynamics.apply, params)
        loss, metrics = self.diffusion.training_losses(model_fn, batch, rng)
        prior_bpd = self.diffusion.prior_bpd(batch).mean()
        metrics["prior_bpd"] = prior_bpd
        return loss, metrics

    @partial(jax.jit, static_argnames=("self", "samples_shape"))
    def samples_fn(self, params, rng, samples_shape) -> jnp.ndarray:
        model_fn = partial(self.dynamics.apply, params)
        samples = self.diffusion.p_sample(model_fn, samples_shape, rng)
        chex.assert_shape(samples, samples_shape)
        return samples.astype(jnp.float32)


class Trainer:
    def __init__(self, config, dataloader, logger: Logger):
        self.config = config
        self.dataloader = dataloader

        # TODO: make these configurable
        self.model = Model(config)
        self.lr_schedule = optax.linear_schedule(
            0.0, self.config.learning_rate, self.config.warmup_steps
        )
        self.optimizer = optax.adam(learning_rate=self.lr_schedule)
        if self.config.grad_clip is not None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.grad_clip), self.optimizer
            )
        self.logger = logger
        logger.info(f"Config: {pprint.pformat(config)}")

    def make_init_state(self, rng) -> TrainState:
        init_params = self.model.make_init_params(rng)
        optimizer_state = self.optimizer.init(init_params)
        return TrainState(step=0, optimizer_state=optimizer_state, params=init_params)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch, rng):
        loss_fn = partial(self.model.loss_fn, rng=rng, batch=batch)
        (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        metrics["grad_norm"] = global_norm(grads)
        metrics["loss"] = loss_value
        updates, new_optimizer_state = self.optimizer.update(
            grads, state.optimizer_state
        )
        new_state = TrainState(
            step=state.step + 1,
            optimizer_state=new_optimizer_state,
            params=optax.apply_updates(state.params, updates),
        )
        return new_state, metrics

    @partial(jax.jit, static_argnames=("self", "num_samples"))
    def generate_samples(self, state, rng, num_samples):
        max_retries = self.config.sample_max_retries
        samples_shape = (num_samples, *self.config.data_shape)

        def _cond(carry):
            i, x, rng = carry
            return jnp.logical_and(
                i < max_retries, jnp.logical_not(jnp.all(jnp.isfinite(x)))
            )

        def _body(carry):
            i, x, rng = carry
            rng, sample_rng = jrnd.split(rng)
            return (
                i + 1,
                self.model.samples_fn(state.params, sample_rng, samples_shape),
                rng,
            )

        _, samples, _ = jax.lax.while_loop(
            _cond,
            _body,
            init_val=(0, jnp.full(samples_shape, jnp.nan, dtype=jnp.float32), rng),
        )

        chex.assert_shape(samples, samples_shape)
        return samples.astype(jnp.int32)

    def train(self):
        os.makedirs(self.config.work_dir, exist_ok=True)
        writer = tensorboard.SummaryWriter(self.config.work_dir)

        rng = jrnd.PRNGKey(self.config.seed)
        rng, init_rng = jrnd.split(rng)
        state = self.make_init_state(init_rng)
        last_log_time = time.time()
        # TODO: load checkpoint if the training procedure is long: see https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html

        for epoch in range(self.config.epochs):
            for batch in self.dataloader:
                rng, step_rng = jrnd.split(rng)
                state, metrics = self.train_step(state, batch, step_rng)
                if state.step % self.config.log_every == 0:
                    self.logger.info(f'Step[{state.step}] Loss: {metrics["loss"]}')
                    for key, value in metrics.items():
                        writer.scalar(f"Train/{key}", value, step=state.step)
                        writer.scalar(
                            f"Train/LR",
                            self.lr_schedule(state.optimizer_state[1][1].count),
                            step=state.step,
                        )
                        if state.step > 0:  # change this if loading from a checkpoint
                            steps_per_sec = self.config.log_every / (
                                time.time() - last_log_time
                            )
                            writer.scalar(
                                "Train/StepsPerSec", steps_per_sec, step=state.step
                            )

        return state
