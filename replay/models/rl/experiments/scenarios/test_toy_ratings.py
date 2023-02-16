from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset
from numpy.random import Generator

from replay.models.rl.experiments.datasets.toy_ratings import generate_clusters
from replay.models.rl.experiments.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from replay.models.rl.experiments.utils.timer import timer, print_with_timestamp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class RandomLogGenerator:
    rng: Generator
    n_users: int
    n_items: int
    n_pairs: int

    def __init__(self, seed: int, n_users: int, n_items: int, n_pairs: int | float):
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_pairs = n_pairs if isinstance(n_pairs, int) else int(n_pairs * n_users * n_items)

    def generate(self) -> pd.DataFrame:
        log_pairs = self.rng.choice(
            self.n_users * self.n_items,
            size=self.n_pairs,
            replace=False
        )
        # timestamps denote the order of interactions, this is not the real timestamp
        timestamps = np.arange(self.n_pairs)
        log_users, log_items = np.divmod(log_pairs, self.n_items)
        log = pd.DataFrame({
            'user_id': log_users,
            'item_id': log_items,
            'timestamp': timestamps,
        })
        log.sort_values(
            ['user_id', 'timestamp'],
            inplace=True,
            ascending=[True, False]
        )
        return log


class RandomEmbeddingsGenerator:
    rng: Generator
    n_dims: int

    def __init__(self, seed: int, n_dims: int):
        self.rng = np.random.default_rng(seed)
        self.n_dims = n_dims

    def generate(self, n: int = None) -> np.ndarray:
        shape = (n, self.n_dims) if n is not None else (self.n_dims,)
        return self.rng.uniform(size=shape)


class RandomClustersEmbeddingsGenerator:
    rng: Generator
    n_dims: int
    intra_cluster_noise_scale: float

    clusters: np.ndarray

    def __init__(
            self, seed: int, n_dims: int, n_clusters: int | list[int],
            intra_cluster_noise_scale: float = 0.05,
            n_dissimilar_dims_required: int = 3,
            min_dim_delta: float = 0.3,
            min_l2_dist: float = 0.1,
            max_generation_tries: int = 10000
    ):
        self.rng = np.random.default_rng(seed)
        self.n_dims = n_dims
        self.intra_cluster_noise_scale = intra_cluster_noise_scale
        self.clusters = generate_clusters(
            self.rng, n_clusters, n_dims,
            n_dissimilar_dims_required=n_dissimilar_dims_required,
            min_dim_delta=min_dim_delta,
            min_l2_dist=min_l2_dist,
            max_tries=max_generation_tries,
        )

    def generate(self, n: int = None) -> np.ndarray:
        if n is None:
            return self.generate_one()
        return np.array([self.generate_one() for _ in range(n)])

    def generate_one(self) -> np.ndarray:
        cluster = self.rng.choice(self.clusters)
        embedding = self.rng.normal(
            loc=cluster, scale=self.intra_cluster_noise_scale, size=(self.n_dims,)
        )
        return np.clip(0, 1, embedding)


class ToyRatingsGenerator:
    metric: str
    positive_ratio: float

    def __init__(self, metric: str, positive_ratio: float):
        self.metric = metric
        self.positive_ratio = positive_ratio

    def similarity(self, users: np.ndarray, items: np.ndarray) -> np.ndarray | float:
        if self.metric == 'l1':
            d = users - items
            return 1 - np.abs(d).mean(axis=-1)
        elif self.metric == 'l2':
            d = users - items
            avg_sq_d = (d ** 2).mean(axis=-1)
            return 1 - np.sqrt(avg_sq_d)
        elif self.metric == 'cosine':
            dot_product = np.sum(users * items, axis=-1)
            users_norm = np.linalg.norm(users, axis=-1)
            items_norm = np.linalg.norm(items, axis=-1)
            return dot_product / (users_norm * items_norm)

    def generate(self, users: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        similarity = self.similarity(users, items)
        relevant_threshold = np.quantile(similarity, 1 - self.positive_ratio, interpolation='lower')
        relevant = similarity >= relevant_threshold

        continuous_ratings = similarity
        discrete_ratings = relevant
        return continuous_ratings, discrete_ratings


class ToyRatingsDataset:
    log: pd.DataFrame
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    def __init__(
            self, global_config: GlobalConfig, source: TConfig,
            embeddings_n_dims: int, user_embeddings: TConfig, item_embeddings: TConfig,
            ratings: TConfig
    ):
        self.global_config = global_config
        self.source = global_config.resolve_object(source)
        self.embeddings_n_dims = embeddings_n_dims

        user_embeddings_generator = global_config.resolve_object(
            user_embeddings, n_dims=embeddings_n_dims
        )
        self.user_embeddings = user_embeddings_generator.generate(self.source.n_users)

        item_embeddings_generator = global_config.resolve_object(
            item_embeddings, n_dims=embeddings_n_dims
        )
        self.item_embeddings = item_embeddings_generator.generate(self.source.n_items)

        self.ratings = ToyRatingsGenerator(**ratings)

    def generate(self):
        self.log = self.source.generate()
        continuous_ratings, discrete_ratings = self.ratings.generate(
            self.log_user_embeddings, self.log_item_embeddings
        )
        self.log['continuous_rating'] = continuous_ratings
        self.log['discrete_rating'] = discrete_ratings

    @property
    def log_user_embeddings(self):
        return self.user_embeddings[self.log['user_id']]

    @property
    def log_item_embeddings(self):
        return self.item_embeddings[self.log['item_id']]

    @property
    def log_continuous_ratings(self):
        return self.log['continuous_rating']

    @property
    def log_discrete_ratings(self):
        return self.log['discrete_rating']


class MdpDatasetBuilder:
    actions: str
    rewards: dict

    def __init__(self, actions: str, rewards: dict):
        self.actions = actions
        self.rewards = rewards

    def build(self, ds):
        observations = np.concatenate((ds.log_user_embeddings, ds.log_item_embeddings), axis=-1)
        actions = self._get_actions(ds)
        rewards = self._get_rewards(ds)
        terminals = self._get_terminals(ds)

        return MDPDataset(
            observations=observations,
            actions=np.expand_dims(actions, axis=-1),
            rewards=rewards,
            terminals=terminals,
        )

    def _get_rewards(self, ds):
        rewards = np.zeros(ds.log.shape[0])
        for name, value in self.rewards.items():
            if name == 'baseline':
                baseline: float = value
                rewards += np.full_like(rewards, baseline)
            elif name == 'continuous':
                weight: float = value
                rewards += ds.log_continuous_ratings.values * weight
            elif name == 'discrete':
                weights = np.array(value)
                rewards = weights[ds.log_discrete_ratings.values.astype(int)]
        return rewards

    def _get_actions(self, ds):
        if self.actions == 'continuous':
            return ds.log_continuous_ratings.values
        elif self.actions == 'discrete':
            return ds.log_discrete_ratings.values
        else:
            raise ValueError(f'Unknown actions type: {self.actions}')

    @staticmethod
    def _get_terminals(ds):
        terminals = np.zeros(ds.log.shape[0])
        terminals[1:] = ds.log['user_id'].values[1:] != ds.log['user_id'].values[:-1]
        terminals[-1] = True
        return terminals


class ToyRatingsExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int

    top_k: int
    epochs: int

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            top_k: int, epochs: int, dataset: TConfig, mdp: TConfig,
            cuda_device: bool | int | None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )

        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        if cuda_device is not None:
            import torch.cuda
            print(f'CUDA available: {torch.cuda.is_available()}')

        self.seed = seed
        self.top_k = top_k
        self.epochs = epochs

        self.dataset = self.config.resolve_object(dataset)
        self.dataset.generate()
        self.train_mdp = MdpDatasetBuilder(**mdp).build(self.dataset)

    def run(self):
        logging.disable(logging.DEBUG)

        self.print_with_timestamp('==> Run')
        from d3rlpy.algos import CQL
        cql = CQL(
            use_gpu=False, batch_size=16,
            # actor_learning_rate=1e-3, critic_learning_rate=3e-3,
        )
        fitter = cql.fitter(
            self.train_mdp,
            n_epochs=self.epochs, verbose=False,
            save_metrics=False, show_progress=False,
        )
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % 10 == 0:
                self._eval_and_print(cql, epoch)

        self.print_with_timestamp('<==')

    def _eval_and_print(self, model, epoch):
        train_prediction = model.predict(self.train_mdp.observations)
        train_loss = np.mean(np.abs(train_prediction - self.train_mdp.actions))
        self.print_with_timestamp(f'Epoch {epoch}: train loss = {train_loss:.4f}')

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)


class TypesResolver(LazyTypeResolver):
    def resolve(self, type_name: str, **kwargs):
        if type_name == 'dataset.toy_ratings':
            return ToyRatingsDataset
        if type_name == 'ds_source.random':
            return RandomLogGenerator
        if type_name == 'embeddings.random':
            return RandomEmbeddingsGenerator
        if type_name == 'embeddings.clusters':
            return RandomClustersEmbeddingsGenerator
        raise ValueError(f'Unknown type: {type_name}')
