import abc
from contextlib import contextmanager
from functools import partial
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jrnd

from d3exp.beta_schedules import get_beta_schedule
from d3exp.config import Config
from d3exp.transitions import get_transition


@contextmanager
def jax_float64_context():
    original_x64_enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield None
    jax.config.update("jax_enable_x64", original_x64_enabled)


class CategoricalDiffusion(metaclass=abc.ABCMeta):
    def __init__(self, config: Config):
        self.num_classes = config.num_classes
        self.num_timesteps = config.num_timesteps
        self.eps = config.eps
        self.hybrid_coeff = config.hybrid_coeff

        # Following the D3PM code, we enforce float64 for beta schedules and the q matrices for better precision
        with jax_float64_context():
            if config.transition_type == "random_double_stochastic":
                self.transition = get_transition(config, rng=jrnd.PRNGKey(config.seed))
            else:
                self.transition = get_transition(config)
            beta_schedule = get_beta_schedule(config)
            self.betas = beta_schedule.get_betas(config.num_timesteps)
            q_one_step_matrices = [
                self.transition._get_transition_matrix(t, self.betas[t])
                for t in range(config.num_timesteps)
            ]
            self.q_onestep_matrices = jnp.stack(q_one_step_matrices, axis=0)
            chex.assert_shape(
                self.q_onestep_matrices,
                (config.num_timesteps, self.num_classes, self.num_classes),
            )

            # q(x_t | x_start)
            def _matmul_with_carry(carry, x):
                product = carry @ x
                return product, product

            self.q_matrices = jax.lax.scan(
                _matmul_with_carry,
                jnp.eye(self.num_classes, dtype=self.q_onestep_matrices.dtype),
                self.q_onestep_matrices,
            )[1]
            chex.assert_shape(
                self.q_matrices,
                (config.num_timesteps, self.num_classes, self.num_classes),
            )

            self.q_onestep_matrices_T = jnp.transpose(
                self.q_onestep_matrices, (0, 2, 1)
            )

    # region loss computation

    @partial(jax.jit, static_argnums=(0, 1))
    def training_losses(self, model_fn, x_start, rng):
        sample_rng, time_rng = jrnd.split(rng)
        t = jrnd.randint(
            time_rng,
            (x_start.shape[0],),
            minval=0,
            maxval=self.num_timesteps,
            dtype=jnp.int32,
        )
        x_t = self.q_sample(x_start, t, sample_rng)

        variational_loss, pred_x_start_logits = self.variational_loss(
            model_fn, x_start, x_t, t
        )
        ce_loss = self.cross_entropy_x_start(x_start, pred_x_start_logits)
        loss = variational_loss + self.hybrid_coeff * ce_loss

        chex.assert_equal_shape([loss, t])
        return loss.mean(axis=0), dict(
            variational_loss=variational_loss.mean(axis=0),
            ce_loss=ce_loss.mean(axis=0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _categorical_kl_logits(self, logits1, logits2):
        out = jax.nn.softmax(logits1 + self.eps, axis=-1) * (
            jax.nn.log_softmax(logits1 + self.eps, axis=-1)
            - jax.nn.log_softmax(logits2 + self.eps, axis=-1)
        )
        return jnp.sum(out, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _categorical_kl_probs(self, probs1, probs2):
        out = probs1 * (jnp.log(probs1 + self.eps) - jnp.log(probs2 + self.eps))
        return jnp.sum(out, axis=-1)

    def _categorical_log_likelihood(self, x, logits):
        chex.assert_type(x, jnp.int32)
        log_probs = jax.nn.log_softmax(logits)
        x_onehot = jax.nn.one_hot(x, logits.shape[-1])
        return jnp.sum(log_probs * x_onehot, axis=-1)

    def variational_loss(self, model_fn, x_start, x_t, t):
        chex.assert_shape(t, (x_start.shape[0],))

        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_format="labels")
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x_t, t)

        kl = self._categorical_kl_logits(true_logits, model_logits)
        chex.assert_equal_shape([kl, x_start])
        kl = kl.mean(axis=tuple(range(1, kl.ndim))) / jnp.log(2)

        decoder_nll = -self._categorical_log_likelihood(x_start, model_logits)
        chex.assert_equal_shape([decoder_nll, x_start])
        decoder_nll = decoder_nll.mean(
            axis=tuple(range(1, decoder_nll.ndim))
        ) / jnp.log(2)

        chex.assert_equal_shape([decoder_nll, kl, t])
        return jnp.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        log_likelihood = self._categorical_log_likelihood(x_start, pred_x_start_logits)
        chex.assert_equal_shape([log_likelihood, x_start])
        return -log_likelihood.mean(
            axis=tuple(range(1, log_likelihood.ndim))
        ) / jnp.log(2)

    # endregion

    # region q sampling

    def prior_bpd(self, x_start):
        # KL(q(x_{T-1} | x_start) || stationary)
        q_probs = self.q_probs(
            x_start,
            t=jnp.full((x_start.shape[0],), self.num_timesteps - 1, dtype=jnp.int32),
        )
        prior_probs = self.transition._stationary_distribution(q_probs.shape)
        chex.assert_equal_shape([q_probs, prior_probs])
        kl_prior = self._categorical_kl_probs(q_probs, prior_probs)
        chex.assert_equal_shape([kl_prior, x_start])
        return kl_prior.mean(axis=tuple(range(1, kl_prior.ndim))) / jnp.log(2)

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "x_start_format",
        ),
    )
    def q_posterior_logits(
        self, x_start, x_t, t, x_start_format: Literal["logits", "labels"]
    ):
        # logits for q(x_{t-1} | x_t, x_start)
        if x_start_format == "logits":
            chex.assert_shape(x_start, (*x_t.shape, self.num_classes))

        elif x_start_format == "labels":
            chex.assert_equal_shape([x_start, x_t])
            chex.assert_type(x_start, jnp.int32)
        else:
            raise ValueError(
                f'x_start_format must be one of "logits" or "labels", got {x_start_format}'
            )

        fact1 = self._array_at_t_x(self.q_onestep_matrices_T, t, x_t, x_format="labels")
        fact2 = self._array_at_t_x(
            self.q_matrices,
            t - 1,
            jax.nn.softmax(x_start, axis=-1) if x_start_format == "logits" else x_start,
            "probs" if x_start_format == "logits" else "labels",
        )
        if x_start_format == "logits":
            x_start_logits = x_start
        else:
            x_start_logits = jnp.log(
                jax.nn.one_hot(x_start, self.num_classes, dtype=x_t.dtype) + self.eps
            )

        out = jnp.log(fact1 + self.eps) + jnp.log(fact2 + self.eps)
        t_broadcast = jnp.expand_dims(t, tuple(range(1, out.ndim)))
        return jnp.where(t_broadcast == 0, x_start_logits, out)

    def _array_at_t_x(self, a, t, x, x_format: Literal["labels", "probs"]):
        """
        a[t, x]

        a: (T, num_classes, num_classes)
        t: (N, )
        x: (N, ...) if t_format == 'labels' else (N, ..., num_classes)

        See the _at and _at_onehot methods in D3PM code
        """
        a_cast = jnp.asarray(a, dtype=jnp.float32)
        if x_format == "labels":
            t_broadcast = jnp.expand_dims(t, axis=tuple(range(1, x.ndim)))
            out = a_cast[t_broadcast, x]
            chex.assert_shape(out, (*x.shape, self.num_classes))
        elif x_format == "probs":
            chex.assert_shape(x, (*x.shape[:-1], self.num_classes))
            if x.ndim == 2:
                x_expanded = jnp.expand_dims(x, axis=1)
                out = jnp.matmul(
                    x_expanded, a_cast[t, ...], precision=jax.lax.Precision.HIGHEST
                )
                out = out.squeeze(axis=1)
            else:
                a_expanded_dims = jnp.expand_dims(
                    a_cast, axis=tuple(range(1, x.ndim - 2))
                )
                chex.assert_shape(t, (x.shape[0],))
                chex.assert_shape(
                    a_expanded_dims[t, ...],
                    (*x.shape[:-2], self.num_classes, self.num_classes),
                )
                out = jnp.matmul(
                    x, a_expanded_dims[t, ...], precision=jax.lax.Precision.HIGHEST
                )
            chex.assert_equal_shape([out, x])
        else:
            raise ValueError(
                f't_format must be one of "labels" or "probs", got {x_format}'
            )
        chex.assert_type(out, jnp.float32)
        return out

    @partial(jax.jit, static_argnums=(0,))
    def q_probs(self, x_start, t):
        # x_start: (N, ...) not one-hot
        return self._array_at_t_x(self.q_matrices, t, x_start, x_format="labels")

    @partial(jax.jit, static_argnums=(0,))
    def q_sample(self, x_start, t, rng):
        # sample from q(x_t | x_start)
        logits = jnp.log(self.q_probs(x_start, t) + self.eps)
        noise = jrnd.uniform(
            rng,
            (*x_start.shape, self.num_classes),
            dtype=logits.dtype,
            minval=jnp.finfo(logits.dtype).tiny,
            maxval=1.0,
        )
        gumbel_noise = -jnp.log(-jnp.log(noise))
        return jnp.argmax(logits + gumbel_noise, axis=-1)

    # endregion

    # region p sampling

    def p_logits(self, model_fn, x, t):
        # logits for p(x_{t-1} | x_t)
        # we are only assuming the model output is logits;
        # TODO: in the future, we can also support discretized truncated logistic distribution, see Appendix A.8 in the D3PM paper
        # Also see line 454-463 https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py#L454-L463
        # TODO: reason about why we can't predict x_{t-1}, but can only predict x_start
        pred_x_start_logits = model_fn(x, t)
        t_broadcast = jnp.expand_dims(t, tuple(range(1, pred_x_start_logits.ndim)))
        model_logits = jnp.where(
            t_broadcast == 0,
            pred_x_start_logits,
            self.q_posterior_logits(pred_x_start_logits, x, t, x_start_format="logits"),
        )
        chex.assert_equal_shape([model_logits, pred_x_start_logits])
        chex.assert_shape(model_logits, (*x.shape, self.num_classes))

        return model_logits, pred_x_start_logits

    def p_sample_one_timestep(self, model_fn, x, t, rng):
        # p(x_{t-1} | x_t)
        noise = jrnd.uniform(
            rng,
            (*x.shape, self.num_classes),
            minval=jnp.finfo(jnp.float32).tiny,
            maxval=1.0,
        )
        gumbel_noise = -jnp.log(-jnp.log(noise))
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x, t)
        chex.assert_equal_shape([model_logits, gumbel_noise])
        t_broadcast = jnp.expand_dims(t, tuple(range(1, model_logits.ndim)))
        sample_logits = jnp.where(
            t_broadcast == 0, model_logits, model_logits + gumbel_noise
        )
        sample = jnp.argmax(sample_logits, axis=-1)
        chex.assert_equal_shape([sample, x])
        return sample

    def p_sample(self, model_fn, shape, rng):
        # sample p(x) from stationary distribution
        init_rng, sample_rng = jrnd.split(rng)
        x_init = self.transition._sample_from_stationary_distribution(shape, init_rng)
        chex.assert_shape(x_init, shape)

        # run a for loop from x_{T-1} to x_start
        def _onestep(i, x):
            t_value = self.num_timesteps - 1 - i
            t = jnp.full((shape[0],), t_value, dtype=jnp.int32)
            x_t, rng = x
            rng, sample_rng = jrnd.split(rng)
            x_t_minus_1 = self.p_sample_one_timestep(model_fn, x_t, t, sample_rng)
            return x_t_minus_1, rng

        final_x, _ = jax.lax.fori_loop(
            0, self.num_timesteps, _onestep, (x_init, sample_rng)
        )
        chex.assert_shape(final_x, shape)
        return final_x

    # endregion
