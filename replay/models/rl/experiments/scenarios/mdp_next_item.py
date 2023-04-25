from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

from replay.models.rl.experiments.datasets.synthetic.relevance import similarity
from replay.models.rl.experiments.run.wandb import get_logger
from replay.models.rl.experiments.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from replay.models.rl.experiments.utils.timer import timer, print_with_timestamp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run
    from d3rlpy.algos import AlgoBase
    from d3rlpy.dataset import MDPDataset


def boosting(relative_value: float | np.ndarray, k: float, softness: float = 1.5) -> float:
    # relative value: value / target value  \in [0, +inf)
    # x = -log(relative_rate)
    #   0 1 +inf  -> +inf 0 -inf
    x = -np.log(relative_value)
    # zero k means "no boosting", that's why we use shifted value.
    K = k + 1

    # relative_rate -> x -> B:
    #   0 -> +inf -> K^tanh(+inf) = K^1 = K
    #   1 -> 0 -> K^tanh(0) = K^0 = 1
    #   +inf -> -inf -> K^tanh(-inf) = K^(-1) = 1 / K
    # higher softness just makes the sigmoid curve smoother; default value is empirically optimized
    return np.power(K, np.tanh(x / softness))


class Embeddings:
    n_dims: int
    users: np.ndarray
    items: np.ndarray

    def __init__(
            self, global_config: GlobalConfig,
            n_users: int, n_items: int,
            n_dims: int, users: TConfig, items: TConfig
    ):
        self.n_dims = n_dims
        self.users = (
            global_config
            .resolve_object(users, n_dims=self.n_dims)
            .generate(n_users)
        )
        item_embeddings_generator = global_config.resolve_object(items, n_dims=self.n_dims)
        self.items = item_embeddings_generator.generate(n_items)
        self.item_clusters = item_embeddings_generator.clusters
        self.n_item_clusters = item_embeddings_generator.n_clusters


class UserState:
    rng: Generator
    user_id: int
    tastes: np.ndarray

    # all projected onto clusters
    # volatile, it also correlates to user's mood
    satiation: np.ndarray
    # changes with reset
    satiation_speed: np.ndarray

    relevance_boosting_k: tuple[float, float]
    metric: str
    embeddings: Embeddings

    def __init__(
            self, user_id: int, embeddings: Embeddings,
            base_satiation: float,
            base_satiation_speed: float | tuple[float, float],
            similarity_metric: str,
            relevance_boosting: tuple[float, float],
            boosting_softness: tuple[float, float],
            rng: Generator
    ):
        self.user_id = user_id
        self.tastes = embeddings.users[user_id]
        self.rng = np.random.default_rng(rng.integers(100_000_000))

        n_clusters = embeddings.n_item_clusters
        self.satiation = base_satiation + self.rng.uniform(size=n_clusters)

        if isinstance(base_satiation_speed, float):
            self.satiation_speed = np.full(embeddings.n_item_clusters, base_satiation_speed)
        else:
            # tuple[float, float]
            base_satiation_speed, k = base_satiation_speed
            k = 1.0 + k
            min_speed, max_speed = 1/k * base_satiation_speed, k * base_satiation_speed
            self.satiation_speed = np.clip(
                self.rng.uniform(min_speed, max_speed, n_clusters),
                0., 1.0
            )

        self.relevance_boosting_k = tuple(relevance_boosting)
        self.boosting_softness = tuple(boosting_softness)
        self.metric = similarity_metric
        self.embeddings = embeddings

    def step(self, item_id: int) -> float:
        # 1) find similarity between item embedding and item clusters
        item_embedding = self.embeddings.items[item_id]
        clusters = self.embeddings.item_clusters
        item_to_cluster_relevance = similarity(item_embedding, clusters, metric=self.metric)
        item_to_cluster_relevance /= item_to_cluster_relevance.sum(-1)

        # 2) increase satiation via similarity and speed
        self.satiation *= 1.0 + item_to_cluster_relevance * self.satiation_speed

        # 3) get item similarity to user preferences and compute boosting
        #       from the aggregate weighted cluster satiation
        base_item_to_user_relevance = similarity(self.tastes, item_embedding, metric=self.metric)
        aggregate_item_satiation = np.sum(self.satiation * item_to_cluster_relevance)

        boosting_k = self.relevance_boosting_k[aggregate_item_satiation > 1.0]
        boosting_softness = self.boosting_softness[aggregate_item_satiation > 1.0]
        relevance_boosting = boosting(
            aggregate_item_satiation, k=boosting_k, softness=boosting_softness
        )

        # 4) compute continuous and discrete relevance as user feedback
        relevance = base_item_to_user_relevance * relevance_boosting

        print(
            f'AggSat {aggregate_item_satiation:.2f} '
            f'| RB {relevance_boosting:.2f} '
            f'| Rel {relevance:.3f}'
        )

        return relevance


class NextItemEnvironment:
    n_users: int
    n_items: int

    embeddings: Embeddings

    max_episode_len: tuple[int, int]
    timestep: int
    state: UserState
    states: list[UserState]
    current_max_episode_len: int

    def __init__(
            self, global_config: GlobalConfig, seed: int,
            n_users: int, n_items: int,
            embeddings: TConfig,
            max_episode_len: int | tuple[int, int],
            user_state: TConfig,
    ):
        self.global_config = global_config
        self.rng = np.random.default_rng(seed)

        self.n_users = n_users
        self.n_items = n_items
        self.embeddings = global_config.resolve_object(
            embeddings | dict(global_config=self.global_config, n_users=n_users, n_items=n_items),
            object_type_or_factory=Embeddings
        )
        self.states = [
            UserState(user_id, embeddings=self.embeddings, rng=self.rng, **user_state)
            for user_id in range(self.n_users)
        ]
        if isinstance(max_episode_len, int):
            self.max_episode_len = (max_episode_len, max_episode_len)
        else:
            # tuple[int, int]: (avg_len, delta)
            self.max_episode_len = (
                max_episode_len[0] - max_episode_len[1],
                max_episode_len[0] + max_episode_len[1]
            )
        self.timestep = 0

    def reset(self, user_id: int = None) -> float:
        if user_id is None:
            user_id = self.rng.integers(self.n_users)

        self.state = self.states[user_id]
        self.timestep = 0
        self.current_max_episode_len = self.rng.integers(*self.max_episode_len)

        print(f'Sat: {self.state.satiation}')
        print(f'SSp: {self.state.satiation_speed}')
        return 0.

    def step(self, item_id: int):
        relevance = self.state.step(item_id)
        self.timestep += 1

        terminated = self.timestep >= self.current_max_episode_len
        if terminated:
            print(f'Sat: {self.state.satiation}')
            print(f'SSp: {self.state.satiation_speed}')
        return relevance, terminated


class MdpNextItemExperiment:
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
            train_test_split: TConfig, negative_samples: TConfig,
            env: TConfig,
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

        self.env = self.config.resolve_object(env, object_type_or_factory=NextItemEnvironment)

        # self.dataset_generator = self.config.resolve_object(dataset)
        # full_dataset = self.dataset_generator.generate()
        # train_dataset = self.dataset_generator.split(full_dataset, **train_test_split)
        # negative_samples = self.dataset_generator.generate_negative_samples(**negative_samples)
        # train_log = pd.concat([train_dataset.log, negative_samples.log], ignore_index=True)
        # train_log.sort_values(
        #     ['user_id', 'timestamp'],
        #     inplace=True,
        #     ascending=[True, False]
        # )
        #
        # mdp_builder = MdpDatasetBuilder(**mdp)
        # self.test_mdp = mdp_builder.build(full_dataset, use_ground_truth=True)
        # self.train_mdp = mdp_builder.build(ToyRatingsDataset(
        #     log=train_log,
        #     user_embeddings=train_dataset.user_embeddings,
        #     item_embeddings=train_dataset.item_embeddings,
        # ), use_ground_truth=False)
        # self.model = self.config.resolve_object(
        #     model | dict(use_gpu=get_cuda_device(cuda_device))
        # )

    def run(self):
        logging.disable(logging.DEBUG)
        self.set_metrics()

        self.print_with_timestamp('==> Run')
        env = self.env
        rng = np.random.default_rng()

        cum_return = 0.
        _ = env.reset()
        while True:
            print(f'[{env.timestep}]')
            relevance, terminated = env.step(3)
            cum_return += relevance
            # print(f'[{env.timestep}] {relevance:.2f}')
            if terminated:
                break

        print(f'Steps: {env.timestep} | CumRel: {cum_return}')
        # fitter = self.model.fitter(
        #     self.train_mdp,
        #     n_epochs=self.epochs, verbose=False,
        #     save_metrics=False, show_progress=False,
        # )
        # for epoch, metrics in fitter:
        #     if epoch == 1 or epoch % self.eval_schedule == 0:
        #         self._eval_and_log(self.model, epoch)

        self.print_with_timestamp('<==')

    def _eval_and_log(self, model, epoch):
        metrics = self._eval_mae(model, self.test_mdp)

        mae, discrete_mae = metrics['mae'], metrics['discrete_mae']
        self.print_with_timestamp(
            f'Epoch {epoch:03}: mae {mae:.4f} | dmae {discrete_mae:.4f}'
        )
        if self.logger:
            metrics |= dict(epoch=epoch)
            self.logger.log(metrics)

    def _eval_mae(self, model: AlgoBase, dataset: MDPDataset):
        batch_size = model.batch_size
        n_splits = dataset.observations.shape[0] // batch_size
        test_prediction = np.concatenate([
            model.predict(batch)
            for batch in np.array_split(dataset.observations, n_splits)
        ])
        mae = np.mean(np.abs(test_prediction - dataset.actions))

        discrete_predictions = self.dataset_generator.relevance.discretize(test_prediction)
        discrete_gt = self.dataset_generator.relevance.discretize(dataset.actions)
        discrete_mae = np.mean(np.abs(discrete_predictions - discrete_gt))
        return {
            'mae': mae,
            'discrete_mae': discrete_mae,
        }

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
            from replay.models.rl.experiments.datasets.synthetic.dataset import \
                ToyRatingsDatasetBuilder
            return ToyRatingsDatasetBuilder
        if type_name == 'ds_source.random':
            from replay.models.rl.experiments.datasets.synthetic.log import RandomLogGenerator
            return RandomLogGenerator
        if type_name == 'embeddings.random':
            from replay.models.rl.experiments.datasets.synthetic.embeddings import \
                RandomEmbeddingsGenerator
            return RandomEmbeddingsGenerator
        if type_name == 'embeddings.clusters':
            from replay.models.rl.experiments.datasets.synthetic.embeddings import \
                RandomClustersEmbeddingsGenerator
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
        if type_name == 'd3rlpy.bc':
            from d3rlpy.algos.bc import BC
            return BC
        raise ValueError(f'Unknown type: {type_name}')


def get_cuda_device(cuda_device: int | None) -> int | bool:
    if cuda_device is not None:
        import torch.cuda
        cuda_available = torch.cuda.is_available()
        print(f'CUDA available: {cuda_available}; device: {cuda_device}')
        if not cuda_available:
            cuda_device = False
    return cuda_device
