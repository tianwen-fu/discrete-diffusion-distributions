import logging
import os
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import jax.random as jrnd

from d3exp.config import get_configs
from d3exp.data import CauchyDataset, MemoryDataLoader, plot_samples
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
    os.makedirs(work_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(work_dir, "log.txt"))
    logging.getLogger().addHandler(logging.StreamHandler())

    for configset_name in args.configs:
        config_set = get_configs(configset_name)
        for config_name, config in config_set.items():
            logging.info(f"Running config {configset_name}/{config_name}")
            config = deepcopy(config)
            config.seed = args.seed
            config.work_dir = os.path.join(work_dir, configset_name, config_name)
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
            trainer = Trainer(config, dataloader, logging.getLogger("trainer"))
            state = trainer.train()
            rng, sample_rng = jrnd.split(rng)
            samples = trainer.generate_samples(state, sample_rng, config.dataset_size)
            plot_samples(config, samples, dataset, config.work_dir)


if __name__ == "__main__":
    main()
