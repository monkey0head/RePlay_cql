from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset
from numpy.random import Generator

from replay.models.rl.experiments.datasets.toy_ratings import generate_clusters
from replay.models.rl.experiments.run.wandb import get_logger
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

    def __init__(
            self, seed: int, n_users: int, n_items: int, n_pairs: int | float,
            duplicates: bool = False
    ):
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_pairs = self._to_pairs(n_pairs)
        self.duplicates = duplicates

    def generate(self, n_pairs: int | float = None, duplicates: bool = None) -> pd.DataFrame:
        n_pairs = self._to_pairs(n_pairs) if n_pairs is not None else self.n_pairs
        duplicates = duplicates if duplicates is not None else self.duplicates

        log_pairs = self.rng.choice(
            self.n_users * self.n_items,
            size=n_pairs,
            replace=duplicates
        )
        # timestamps denote the order of interactions, this is not the real timestamp
        timestamps = self.rng.uniform(size=n_pairs)
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

    def _to_pairs(self, n_pairs: int | float):
        return n_pairs if isinstance(n_pairs, int) else int(n_pairs * self.n_users * self.n_items)


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


class RelevanceCalculator:
    metric: str
    positive_ratio: float

    def __init__(self, metric: str, positive_ratio: float):
        self.metric = metric
        self.positive_ratio = positive_ratio
        self.relevant_threshold = None

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

    def calculate(self, users: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        similarity = self.similarity(users, items)
        if self.relevant_threshold is None:
            self.relevant_threshold = np.quantile(
                similarity, 1 - self.positive_ratio, interpolation='lower'
            )
        relevant = similarity >= self.relevant_threshold

        continuous_relevance = similarity
        discrete_relevance = relevant.astype(int)
        return continuous_relevance, discrete_relevance


@dataclasses.dataclass
class ToyRatingsDataset:
    log: pd.DataFrame
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

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

    @property
    def log_ground_truth(self):
        return self.log['ground_truth']


class ToyRatingsDatasetBuilder:
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    def __init__(
            self, global_config: GlobalConfig, seed: int, source: TConfig,
            embeddings_n_dims: int, user_embeddings: TConfig, item_embeddings: TConfig,
            ratings: TConfig
    ):
        self.global_config = global_config
        self.rng = np.random.default_rng(seed)
        self.source = global_config.resolve_object(source)
        self.embeddings_n_dims = embeddings_n_dims

        user_embeddings_generator = global_config.resolve_object(
            user_embeddings, n_dims=embeddings_n_dims
        )
        item_embeddings_generator = global_config.resolve_object(
            item_embeddings, n_dims=embeddings_n_dims
        )
        self.user_embeddings = user_embeddings_generator.generate(self.source.n_users)
        self.item_embeddings = item_embeddings_generator.generate(self.source.n_items)
        self.relevance = RelevanceCalculator(**ratings)

    def generate(self) -> ToyRatingsDataset:
        dataset = ToyRatingsDataset(
            self.source.generate(),
            user_embeddings=self.user_embeddings, item_embeddings=self.item_embeddings
        )
        continuous_ratings, discrete_ratings = self.relevance.calculate(
            dataset.log_user_embeddings, dataset.log_item_embeddings
        )
        dataset.log['continuous_rating'] = continuous_ratings
        dataset.log['discrete_rating'] = discrete_ratings
        dataset.log['gt_continuous_rating'] = continuous_ratings
        dataset.log['gt_discrete_rating'] = discrete_ratings
        dataset.log['ground_truth'] = True
        return dataset

    def generate_augmentations(self, n_pairs: int | float) -> ToyRatingsDataset:
        dataset = ToyRatingsDataset(
            self.source.generate(n_pairs=n_pairs, duplicates=True),
            user_embeddings=self.user_embeddings, item_embeddings=self.item_embeddings
        )
        continuous_ratings, discrete_ratings = self.relevance.calculate(
            dataset.log_user_embeddings, dataset.log_item_embeddings
        )
        rand_order = self.rng.permutation(continuous_ratings.shape[0])
        dataset.log['continuous_rating'] = continuous_ratings[rand_order]
        dataset.log['discrete_rating'] = discrete_ratings[rand_order]
        dataset.log['gt_continuous_rating'] = continuous_ratings
        dataset.log['gt_discrete_rating'] = discrete_ratings
        dataset.log['ground_truth'] = False
        return dataset

    def split(self, dataset: ToyRatingsDataset, split_by, train: float) -> ToyRatingsDataset:
        n_train_users = int(train * self.source.n_users)
        train_users = self.rng.choice(self.source.n_users, size=n_train_users, replace=False)
        return ToyRatingsDataset(
            dataset.log[dataset.log['user_id'].isin(train_users)].copy(),
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings
        )


class MdpDatasetBuilder:
    actions: str
    rewards: dict

    def __init__(self, actions: str, rewards: dict):
        self.actions = actions
        self.rewards = rewards

    def build(self, ds: ToyRatingsDataset) -> MDPDataset:
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
                baseline = np.array(value)
                rewards += baseline[ds.log_ground_truth.values.astype(int)]
            elif name == 'continuous':
                weight: float = value
                rewards[ds.log_ground_truth] += (
                        ds.log_continuous_ratings[ds.log_ground_truth] * weight
                )
            elif name == 'discrete':
                weights = np.array(value)
                rewards[ds.log_ground_truth] = weights[
                    ds.log_discrete_ratings[ds.log_ground_truth]
                ]
            elif name == 'continuous_error':
                weight: float = value
                err = np.abs(
                    ds.log_continuous_ratings[~ds.log_ground_truth]
                    - ds.log[~ds.log_ground_truth]['gt_continuous_rating']
                )
                rewards[~ds.log_ground_truth] -= err * weight
            elif name == 'discrete_error':
                weight: float = value
                err = np.abs(
                    ds.log_discrete_ratings[~ds.log_ground_truth]
                    - ds.log[~ds.log_ground_truth]['gt_discrete_rating']
                )
                rewards[~ds.log_ground_truth] -= err * weight
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
    eval_schedule: int

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            top_k: int, epochs: int, dataset: TConfig, mdp: TConfig, model: TConfig,
            train_test_split: TConfig, augmentations: TConfig,
            log: bool, eval_schedule: int,
            cuda_device: bool | int | None,
            project: str = None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )
        self.logger = get_logger(config, log=log, project=project)

        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.seed = seed
        self.top_k = top_k
        self.epochs = epochs
        self.eval_schedule = eval_schedule

        self.dataset_generator = self.config.resolve_object(dataset)
        full_dataset = self.dataset_generator.generate()
        train_dataset = self.dataset_generator.split(full_dataset, **train_test_split)
        augmentations = self.dataset_generator.generate_augmentations(**augmentations)
        train_log = pd.concat([train_dataset.log, augmentations.log], ignore_index=True)
        train_log.sort_values(
            ['user_id', 'timestamp'],
            inplace=True,
            ascending=[True, False]
        )

        mdp_builder = MdpDatasetBuilder(**mdp)
        self.test_mdp = mdp_builder.build(full_dataset)
        self.train_mdp = mdp_builder.build(ToyRatingsDataset(
            log=train_log,
            user_embeddings=train_dataset.user_embeddings,
            item_embeddings=train_dataset.item_embeddings,
        ))
        self.model = self.config.resolve_object(
            model | dict(use_gpu=get_cuda_device(cuda_device))
        )

    def run(self):
        logging.disable(logging.DEBUG)
        self.set_metrics()

        self.print_with_timestamp('==> Run')
        fitter = self.model.fitter(
            self.train_mdp,
            n_epochs=self.epochs, verbose=False,
            save_metrics=False, show_progress=False,
        )
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % self.eval_schedule == 0:
                self._eval_and_log(self.model, epoch)

        self.print_with_timestamp('<==')

    def _eval_and_log(self, model, epoch):
        batch_size = model.batch_size
        n_splits = self.test_mdp.observations.shape[0] // batch_size

        test_prediction = np.vstack([
            model.predict(batch)
            for batch in np.array_split(self.test_mdp.observations, n_splits)
        ])
        test_loss = np.mean(np.abs(test_prediction - self.test_mdp.actions))
        self.print_with_timestamp(f'Epoch {epoch}: test loss = {test_loss:.4f}')
        if self.logger:
            self.logger.log({
                'epoch': epoch,
                'mae': test_loss,
            })

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)

    def set_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('mae', step_metric='epoch')


class TypesResolver(LazyTypeResolver):
    def resolve(self, type_name: str, **kwargs):
        if type_name == 'dataset.toy_ratings':
            return ToyRatingsDatasetBuilder
        if type_name == 'ds_source.random':
            return RandomLogGenerator
        if type_name == 'embeddings.random':
            return RandomEmbeddingsGenerator
        if type_name == 'embeddings.clusters':
            return RandomClustersEmbeddingsGenerator
        if type_name == 'd3rlpy.cql':
            from d3rlpy.algos import CQL
            return CQL
        if type_name == 'd3rlpy.sac':
            from d3rlpy.algos import SAC
            return SAC
        if type_name == 'd3rlpy.ddpg':
            from d3rlpy.algos import DDPG
            return DDPG
        if type_name == 'd3rlpy.discrete_cql':
            from d3rlpy.algos import DiscreteCQL
            return DiscreteCQL
        if type_name == 'd3rlpy.sdac':
            from replay.models.rl.sdac.sdac import SDAC
            return SDAC
        if type_name == 'd3rlpy.discrete_sac':
            from d3rlpy.algos import DiscreteSAC
            return DiscreteSAC
        raise ValueError(f'Unknown type: {type_name}')


def get_cuda_device(cuda_device: int | None) -> int | bool:
    if cuda_device is not None:
        import torch.cuda
        cuda_available = torch.cuda.is_available()
        print(f'CUDA available: {cuda_available}; device: {cuda_device}')
        if not cuda_available:
            cuda_device = False
    return cuda_device
