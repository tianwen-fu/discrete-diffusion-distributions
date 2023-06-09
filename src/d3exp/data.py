import jax.numpy as jnp
import jax.random as jrnd
import numpy as np


class CauchyDist:
    def __init__(self, mode: np.ndarray, tau: np.ndarray, n_classes: int):
        self.mode = mode.reshape(1, -1)
        self.tau = tau.reshape(1, -1)
        self.n_classes = n_classes

    def __call__(self, x: np.ndarray[float]):
        denominator = (
            1
            + np.square((x.reshape(-1, 1) - self.mode) / self.n_classes) / self.tau**2
        )
        unnormalized = (1.0 / self.tau / denominator).sum(axis=1)
        return unnormalized / unnormalized.sum()


class CauchyDataset:
    def __init__(self, n_mode: int, tau: float, n_classes: int, size: int):
        self.size = size
        self.classes = np.arange(n_classes)
        self.mode = (np.arange(n_mode) + 1) * n_classes // (n_mode + 1)
        self.tau = np.full(self.mode.shape, tau)

        density_fn = CauchyDist(mode=self.mode, tau=self.tau, n_classes=n_classes)
        self.p = density_fn(self.classes)
        self.data = np.random.choice(
            self.classes, size=self.size, replace=True, p=self.p
        )

    def __getitem__(self, index):
        return self.data[index].reshape(1)

    def __len__(self):
        return len(self.data)


class MemoryDataLoader:
    def __init__(self, data, batch_size, key, shuffle=False):
        self.data = data  # (N, ...)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key
        assert self.data.shape[0] % self.batch_size == 0

    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def __iter__(self):
        if self.shuffle:
            self.key, key = jrnd.split(self.key)
            batch_order = jrnd.permutation(key, self.data.shape[0]).reshape(
                -1, self.batch_size
            )
        else:
            batch_order = jnp.arange(self.data.shape[0]).reshape(-1, self.batch_size)
        for batch_idx in batch_order:
            yield self.data[batch_idx, ...]
