import abc

import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import scipy
from flax.metrics import tensorboard

from d3exp.config import Config


class DiffusionTransition(metaclass=abc.ABCMeta):
    # following traditions in D3PM codebase, all the matrices shall be in float64 for better precision
    @abc.abstractmethod
    def _get_transition_matrix(self, t, beta_t):
        pass

    @abc.abstractmethod
    def _stationary_distribution(self, prob_shape):
        pass

    @abc.abstractmethod
    def _sample_from_stationary_distribution(self, x_init_shape, rng):
        pass


class TransitionWithUniformStationary(DiffusionTransition):
    # same stationary
    def _stationary_distribution(self, prob_shape):
        return jnp.ones(prob_shape, dtype=jnp.float32) / self.num_classes

    def _sample_from_stationary_distribution(self, x_init_shape, rng):
        return jrnd.randint(rng, x_init_shape, 0, self.num_classes, dtype=jnp.int32)


# Uniform, Gaussian, and Absorbing transitions are from the D3PM code
class UniformTransition(TransitionWithUniformStationary):
    def __init__(self, num_classes, transition_bands=None):
        self.num_classes = num_classes
        self.transition_bands = transition_bands

    def _get_transition_matrix(self, t, beta_t):
        r"""
        With prob beta_t / num_classes, transition to other states (in the same band, or any state if transition_bands is None)
        otherwise, stay in the same state

        Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
                 1 - \sum_{l \neq i} Q_{il}    if i==j.
                 0                             else.
        """
        transition_prob = beta_t / self.num_classes
        if self.transition_bands is None:
            matrix = np.full(
                (self.num_classes, self.num_classes), transition_prob, dtype=np.float64
            )
            matrix[np.diag_indices_from(matrix)] = 0
        else:
            matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
            filler = np.full((self.num_classes - 1,), transition_prob, dtype=np.float64)
            for k in range(1, self.transition_bands + 1):
                mat += np.diag(filler, k=k)
                mat += np.diag(filler, k=-k)
                filler = filler[:-1]  # drop one item

        diag = 1.0 - np.sum(matrix, axis=1)
        matrix += np.diag(diag)
        return matrix


class GaussianTransition(TransitionWithUniformStationary):
    def __init__(self, num_classes, transition_bands=None):
        self.num_classes = num_classes
        self.transition_bands = transition_bands

    def _get_transition_matrix(self, t, beta_t):
        r"""
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                  1 - \sum_{l \neq i} Q_{il} if i==j.
                  0                          else.
        """
        transition_bands = self.transition_bands or self.num_classes - 1

        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = np.linspace(
            start=0.0,
            stop=self.num_classes,
            num=self.num_classes,
            endpoint=True,
            dtype=np.float64,
        )
        values = values * 2.0 / (self.num_classes - 1.0)
        values = values[: transition_bands + 1]
        values = -values * values / beta_t

        values = np.concatenate([values[:0:-1], values], axis=0)
        values = scipy.special.softmax(values, axis=0)
        values = values[transition_bands:]
        for k in range(1, transition_bands + 1):
            off_diag = np.full(
                shape=(self.num_classes - k,), fill_value=values[k], dtype=np.float64
            )
            matrix += np.diag(off_diag, k=k)
            matrix += np.diag(off_diag, k=-k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1.0 - matrix.sum(1)
        matrix += np.diag(diag, k=0)

        return matrix


class AbsorbingTransition(DiffusionTransition):
    def __init__(self, num_classes, stationary_state: int):
        self.num_classes = num_classes
        if stationary_state > num_classes:
            raise ValueError("stationary_state must be a valid class index")
        self.stationary_state = stationary_state

    def _get_transition_matrix(self, t, beta_t):
        """
        Q_ij = beta_t if j == stationary_state
               1-beta_t if i == j
               0 otherwise
        """
        matrix = np.diag(np.full(self.num_classes, 1.0 - beta_t, dtype=np.float64))
        matrix[:, self.stationary_state] += beta_t
        return matrix

    def _stationary_distribution(self, prob_shape):
        labels = jnp.full(prob_shape[:-1], self.stationary_state, dtype=jnp.int32)
        return jax.nn.one_hot(labels, self.num_classes, axis=-1, dtype=jnp.float32)

    def _sample_from_stationary_distribution(self, x_init_shape, rng):
        return jnp.full(x_init_shape, self.stationary_state, dtype=jnp.int32)


# TODO: absorbing + uniform? (beta -> stationary, alpha -> other)


class RandomDoublyStochasticTransition(TransitionWithUniformStationary):
    # This is ours!!!
    @staticmethod
    def generate_doubly_stochastic(target_size, rng):
        # code adapted from https://www.mathworks.com/matlabcentral/answers/53957-is-there-a-better-way-to-randomly-generate-a-doubly-stochastic-matrix
        multiplies_to_validate_connected = 1000
        while True:
            # generate a doubly stochastic matrix
            matrix = jnp.zeros((target_size, target_size))
            rng, c_rng = jrnd.split(rng)
            c = jrnd.uniform(c_rng, shape=(target_size,), minval=0.1, maxval=1.0)
            c /= c.sum()
            I = jnp.eye(target_size, dtype=jnp.float64)
            for i in range(target_size):
                rng, iter_rng = jrnd.split(rng)
                idx = jrnd.permutation(iter_rng, target_size)
                P = I[idx, :]
                matrix = matrix + c[i] * P

            # check if this is connected, so that we can make sure the stationary distribution is uniform distribution
            mat_prod = matrix
            for _ in range(multiplies_to_validate_connected):
                mat_prod = mat_prod @ matrix
                if jnp.all(mat_prod > 0):
                    return matrix

    def __init__(self, num_classes, rng, beta_max: int):
        self.num_classes = num_classes
        self.rng = rng
        self.beta_max = beta_max
        transition_matrix = self.generate_doubly_stochastic(num_classes, rng)
        transition_matrices = [
            jnp.linalg.matrix_power(
                transition_matrix, beta
            )  # because jax somehow does not support batched matrix power
            for beta in range(beta_max)
        ]
        transition_matrices = jnp.stack(transition_matrices, axis=0)
        transition_matrices[0, ...] = jnp.nan  # to make sure we don't use this matrix
        self.transition_matrices = transition_matrices

    def _get_transition_matrix(self, t, beta_t):
        # NOTE: Make sure your beta is in [1, beta_max]!!
        # as an example with seed = 0, beta_max = 8 would become a uni

        beta_casted = jnp.floor(beta_t).astype(jnp.int32)
        return jnp.where(
            (beta_casted > 0) & (beta_casted < self.beta_max),
            self.transition_matrices[beta_casted, ...],
            jnp.nan,
        )


TRANSITION_CLASSES = dict(
    uniform=UniformTransition,
    gaussian=GaussianTransition,
    random_double_stochastic=RandomDoublyStochasticTransition,
    absorbing=AbsorbingTransition,
)


def get_transition(config: Config):
    return TRANSITION_CLASSES[config.transition_type](
        num_classes=config.num_classes, **config.transition_kwargs
    )
