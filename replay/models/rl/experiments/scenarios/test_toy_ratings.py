from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.random import Generator

from replay.models.rl.experiments.utils.config import TConfig, GlobalConfig, resolve_init_params
from replay.models.rl.experiments.utils.timer import timer, print_with_timestamp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class ToyRatingsDatasetGenerator:
    rng: Generator
    embeddings_ndims: int
    n_users: int
    n_items: int
    n_pairs: int
    distance: str
    optimism_bias_power: float

    log: pd.DataFrame
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    def __init__(
            self, seed: int, embeddings_ndims: int, n_users: int, n_items: int,
            n_pairs: int | float, distance: str, optimism_bias_power: float = 1.0
    ):
        self.rng = np.random.default_rng(seed)
        self.embeddings_ndims = embeddings_ndims
        self.n_users = n_users
        self.n_items = n_items
        self.n_pairs = n_pairs if isinstance(n_pairs, int) else int(n_pairs * n_users * n_items)
        self.distance = distance
        self.optimism_bias_power = optimism_bias_power

    def generate(self):
        self.generate_log()
        self.generate_embeddings()
        self.generate_ratings()

    def generate_embeddings(self):
        self.user_embeddings = np.clip(
            0, 1, self.rng.uniform(size=(self.n_users, self.embeddings_ndims))
        )
        self.item_embeddings = np.clip(
            0, 1, self.rng.uniform(size=(self.n_items, self.embeddings_ndims))
        )

    @property
    def log_user_embeddings(self):
        return self.user_embeddings[self.log['user_id']]

    @property
    def log_item_embeddings(self):
        return self.item_embeddings[self.log['item_id']]

    def similarity(self, users: np.ndarray, items: np.ndarray) -> np.ndarray | float:
        if self.distance == 'l1':
            d = users - items
            return 1 - np.abs(d).mean(axis=-1)
        elif self.distance == 'l2':
            d = users - items
            avg_sq_d = (d ** 2).mean(axis=-1)
            return 1 - np.sqrt(avg_sq_d)
        elif self.distance == 'cosine':
            dot_product = np.sum(users * items, axis=-1)
            users_norm = np.linalg.norm(users, axis=-1)
            items_norm = np.linalg.norm(items, axis=-1)
            return dot_product / (users_norm * items_norm)

    def generate_ratings(self):
        like_prob = self.similarity(self.log_user_embeddings, self.log_item_embeddings)
        # if optimism_bias_power > 1, then the user is more likely to like the item
        like_prob = like_prob ** (1.0 / self.optimism_bias_power)

        self.log['rating'] = self.rng.binomial(1, like_prob).flatten()

    def generate_log(self):
        train_pairs = self.rng.choice(
            self.n_users * self.n_items,
            size=self.n_pairs,
            replace=False
        )
        log_users, log_items = np.divmod(train_pairs, self.n_items)
        timestamps = np.arange(self.n_pairs)
        self.log = pd.DataFrame({
            'user_id': log_users,
            'item_id': log_items,
            'timestamp': timestamps,
        })


class ToyRatingsExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int

    top_k: int
    epochs: int

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            top_k: int, epochs: int, dataset_source: TConfig,
            cuda_device: bool | int | None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=None
        )
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        if cuda_device is not None:
            import torch.cuda
            print(f'CUDA available: {torch.cuda.is_available()}')

        self.seed = seed
        self.top_k = top_k
        self.epochs = epochs

        dataset_source = resolve_init_params(dataset_source, seed=self.seed)
        self.dataset_generator = ToyRatingsDatasetGenerator(**dataset_source)
        self.dataset_generator.generate()
        print(len(self.dataset_generator.log))
        print(self.dataset_generator.log['rating'].sum() / self.dataset_generator.n_pairs)
        # with np.printoptions(precision=1):
        #     print(self.dataset_generator.user_embeddings)
        #     print(self.dataset_generator.item_embeddings)

    def run(self):
        self.print_with_timestamp('==> Run')
        self.print_with_timestamp('<==')

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)
