import os

import numpy as np
from matplotlib import pyplot as plt

from d3exp.config import Config
from d3exp.data import CauchyDataset


def samples_to_frequency(samples: np.ndarray, num_classes: int) -> np.ndarray:
    class_idx = np.arange(num_classes).reshape(-1, 1)
    counts = (
        (np.asarray(samples).reshape(1, -1) == class_idx).astype(np.float32).sum(axis=1)
    )
    assert counts.shape == (num_classes,)
    return counts / counts.sum()


def plot_samples(
    config: Config,
    samples,
    dataset: CauchyDataset,
    save_folder: str,
    exp_name: str,
    draw_pmf: bool = False,
):
    fig = plt.figure(figsize=(3, 3), dpi=700)
    plt.plot(
        np.arange(config.num_classes),
        samples_to_frequency(dataset.data, config.num_classes),
        color="orange",
        label="GT",
        alpha=0.7,
    )
    plt.ylim([-0.1, 1.1])
    plt.savefig(
        os.path.join(save_folder, f"{exp_name}_dataset.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.plot(
        np.arange(config.num_classes),
        samples_to_frequency(samples, config.num_classes),
        color="blue",
        label=r"$p_\theta$",
        alpha=0.5,
        linestyle="--",
    )
    if draw_pmf:
        plt.plot(
            np.arange(config.num_classes),
            dataset.p,
            color="red",
            label="PMF",
            alpha=0.7,
        )
    plt.xlabel("$x$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(
        os.path.join(save_folder, f"{exp_name}_samples.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def empirical_kl(config: Config, data_p, data_q):
    p = samples_to_frequency(data_p.reshape(-1), config.num_classes)
    q = samples_to_frequency(data_q.reshape(-1), config.num_classes)
    assert p.shape == q.shape
    return (p * (np.log2(p + config.eps) - np.log2(q + config.eps))).sum()
