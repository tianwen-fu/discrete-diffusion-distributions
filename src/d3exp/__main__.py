import logging
import os
import pickle
import pprint
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import jax.random as jrnd
import numpy as np

from d3exp.config import get_configs
from d3exp.data import CauchyDataset, MemoryDataLoader
from d3exp.evaluation import empirical_kl, plot_samples
from d3exp.trainer import Trainer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("configs", nargs="+", type=str)
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seed = args.seed
    rng = jrnd.PRNGKey(seed)
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = os.path.join(
            os.curdir,
            "results",
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    logging.basicConfig(level=logging.INFO)

    for configset_name in args.configs:
        eval_metrics = {}
        logger = logging.getLogger(configset_name)
        current_work_dir = os.path.join(work_dir, configset_name)
        os.makedirs(current_work_dir, exist_ok=True)
        logger.addHandler(
            logging.FileHandler(os.path.join(current_work_dir, "log.txt"))
        )
        config_set = get_configs(configset_name)
        for config_name, config in config_set.items():
            logger.info(f"Running config {configset_name}/{config_name}")
            config = deepcopy(config)
            config.seed = args.seed
            config.work_dir = os.path.join(current_work_dir, config_name)
            os.makedirs(config.work_dir, exist_ok=True)

            rng, data_rng = jrnd.split(rng)
            dataset = CauchyDataset(
                n_mode=config.n_modes,
                tau=config.data_tau,
                n_classes=config.num_classes,
                size=config.dataset_size,
            )
            dataloader = MemoryDataLoader(
                dataset.data.reshape(-1),
                batch_size=config.batch_size,
                key=data_rng,
                shuffle=True,
            )
            trainer = Trainer(config, dataloader, logger)
            state, metrics = trainer.train()
            rng, sample_rng = jrnd.split(rng)
            samples = np.asarray(
                trainer.generate_samples(state, sample_rng, config.dataset_size)
            )
            plot_samples(
                config,
                samples,
                dataset,
                os.path.join(work_dir, configset_name),
                config_name,
            )
            np.savez(
                os.path.join(config.work_dir, "samples.npz"),
                samples=samples.reshape(-1),
                dataset=dataset.data.reshape(-1),
            )
            eval_metrics[configset_name + "/" + config_name] = {
                key: np.asarray(value) for key, value in metrics.items()
            }
            eval_metrics[configset_name + "/" + config_name].update(
                dict(
                    kl_samples_dataset=empirical_kl(config, samples, dataset.data),
                    kl_dataset_samples=empirical_kl(config, dataset.data, samples),
                )
            )
        logger.info(f"Metrics: {pprint.pformat(eval_metrics)}")
        with open(os.path.join(current_work_dir, "metrics.pkl"), "wb") as f:
            pickle.dump(eval_metrics, f)


if __name__ == "__main__":
    main()
