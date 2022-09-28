import datetime
import logging
import os
import time
import warnings
from argparse import ArgumentParser
from math import floor
from typing import Union

import numpy as np
import pandas as pd
import psutil
import torch
import tqdm
from optuna.exceptions import ExperimentalWarning
from pyspark.sql import functions as sf, SparkSession, DataFrame

from replay.data_preparator import DataPreparator, Indexer
from replay.experiment import Experiment
from replay.metrics import HitRate, NDCG, MAP, MRR, Coverage, Surprisal
from replay.models import Recommender
from replay.session_handler import State, get_spark_session
from replay.splitters import DateSplitter
from replay.utils import get_log_info


class RatingsDataset:
    name: str
    category: str

    log: DataFrame
    binary_train: DataFrame
    pos_binary_train: DataFrame
    raw_train: DataFrame
    test: DataFrame
    test_users: DataFrame

    def __init__(self, name: str, test_ratio: float):
        self.name, self.category = name.split('.')

        if self.name == 'MovieLens':
            from rs_datasets import MovieLens
            ds = MovieLens(version=self.category)
        elif self.name == 'Amazon':
            from rs_datasets import Amazon
            ds = Amazon(category=self.category)
        else:
            raise KeyError()

        col_mapping = {
            'user_id': 'user_id',
            'item_id': 'item_id',
            'relevance': 'rating',
            'timestamp': 'timestamp'
        }
        pd_log = ds.ratings

        data_preparator = DataPreparator()
        log = data_preparator.transform(columns_mapping=col_mapping, data=pd_log)

        indexer = Indexer()
        indexer.fit(users=log.select('user_id'), items=log.select('item_id'))
        self.log = indexer.transform(log)

        log.unpersist()
        self._build_train_test(test_ratio=test_ratio)

    def _build_train_test(self, test_ratio: float):
        log = self.log.cache()

        # will consider ratings >= 3 as positive feedback and negative otherwise
        binary_log = log.withColumn(
            'relevance',
            sf
            .when(sf.col('relevance') >= sf.lit(3), sf.lit(1.0))
            .otherwise(sf.lit(0.))
        ).cache()

        # train/test split
        date_splitter = DateSplitter(
            test_start=test_ratio,
            drop_cold_items=True, drop_cold_users=True, drop_zero_rel_in_test=True
        )
        binary_train, test = date_splitter.split(binary_log)
        pos_binary_train = binary_train.filter(sf.col('relevance') > sf.lit(0)).cache()

        test_start = test.agg(sf.min('timestamp')).collect()[0][0]
        raw_train = log.filter(sf.col('timestamp') < test_start).cache()

        # rew_train = (
        #     log
        #     .filter(sf.col('timestamp') < test_start)
        #     .withColumn(
        #         'relevance',
        #         sf
        #         .when(sf.col('relevance') == sf.lit(1.0), sf.lit(-1.0))
        #         .when(sf.col('relevance') == sf.lit(2.0), sf.lit(-0.3))
        #         .when(sf.col('relevance') == sf.lit(3.0), sf.lit(0.25))
        #         .when(sf.col('relevance') == sf.lit(4.0), sf.lit(0.7))
        #         .when(sf.col('relevance') == sf.lit(5.0), sf.lit(1.0))
        #     )
        #     .cache()
        # )
        test_users = test.select('user_idx').distinct().cache()

        # if self.rating_based_reward:
        #     # raw relevance transformed into rewards-like
        #     cql_train = rew_train
        #     if self.take_positive_events:
        #         cql_train = rew_train.filter(sf.col('relevance') > 0).cache()
        # # prepare train data for CQL
        # else:
        #     # binarized relevance
        #     cql_train = pos_binary_train if self.take_positive_events else binary_train

        self.binary_train = binary_train
        self.pos_binary_train = pos_binary_train
        self.raw_train = raw_train
        self.test = test
        self.test_users = test_users

    @property
    def fullname(self):
        return f'{self.name}.{self.category}'


class BareRatingsRunner:
    init_time: float = time.time()
    gpu: Union[bool, int]
    seed: int
    k: int
    ks: list[int]

    epochs: list[int]
    action_randomization_scale: float
    use_negative_events: bool
    rating_based_reward: bool
    reward_top_k: bool

    logger: logging.Logger
    dataset: RatingsDataset
    experiment: Experiment
    models: dict[str, tuple[Recommender, DataFrame]]

    def __init__(
            self, *,
            dataset_name: str,
            partitions: float, memory: float, gpu: int,
            algorithms: list[str], epochs: list[int], label: str,
            k: Union[int, list[int]], test_ratio: float,
            action_randomization_scale: float, use_negative_events: bool,
            rating_based_reward: bool, reward_top_k: bool,
            seed: int = None
    ):
        self.logger = logging.getLogger("replay")

        init_spark_session(memory, partitions)
        self.print_time('===> Spark initialized')

        self.gpu = gpu if gpu >= 0 and torch.cuda.is_available() else False

        self.dataset = RatingsDataset(dataset_name, test_ratio=test_ratio)
        self.logger.info(msg='train info:\n\t' + get_log_info(self.dataset.binary_train))
        self.logger.info(msg='test info:\n\t' + get_log_info(self.dataset.test))
        self.print_time('===> Dataset prepared')

        self.seed = seed if seed is not None else np.random.default_rng().integers(1_000_000)
        self.ks = list(sorted(k)) if isinstance(k, list) else [k]
        self.k = self.ks[-1]
        self.epochs = list(sorted(epochs))

        self.action_randomization_scale = action_randomization_scale
        self.use_negative_events = use_negative_events
        self.rating_based_reward = rating_based_reward
        self.reward_top_k = reward_top_k

        self.label = label

        self.experiment = self.build_experiment()
        self.print_time('===> Experiment initialized')
        self.models = self.build_models(algorithms=algorithms)
        self.print_time('===> Models initialized')

    def print_time(self, text: str):
        self.logger.info(msg=f'{text}: \t{time.time() - self.init_time:.3f}')

    def run(self):
        results_label = f'{self.label}.{self.dataset.fullname}.md'
        self.logger.info(msg=f'Results are saved to \n\t{results_label}')

        for model_name in tqdm.tqdm(self.models.keys(), desc='Model'):
            model, train = self.models[model_name]
            self.logger.info(msg='{} started'.format(model_name))

            self.fit_predict_add_res(
                model_name, model, self.experiment,
                train=train, top_k=self.k, test_users=self.dataset.test_users
            )
            print(
                self.experiment.results[[
                    f'NDCG@{self.k}', f'MRR@{self.k}', f'Coverage@{self.k}', 'fit_time'
                ]].sort_values(f'NDCG@{self.k}', ascending=False)
            )

            results_md = self.experiment.results.sort_values(
                f'NDCG@{self.k}', ascending=False
            ).to_markdown()
            with open(results_label, 'w') as text_file:
                text_file.write(results_md)

        self.print_time('===> Experiment finished')

    def fit_predict_add_res(
            self,
            name: str, model: Recommender, experiment: Experiment,
            train: pd.DataFrame, top_k: int, test_users: pd.DataFrame
    ):
        """
        Run fit_predict for the `model`, measure time on fit_predict and evaluate metrics
        """
        start_time = time.time()

        model.fit(log=train)
        fit_time = time.time() - start_time

        pred = model.predict(log=train, k=top_k, users=test_users).cache()
        predict_time = time.time() - start_time - fit_time

        experiment.add_result(name, pred)
        metric_time = time.time() - start_time - fit_time - predict_time

        experiment.results.loc[name, 'fit_time'] = fit_time
        experiment.results.loc[name, 'predict_time'] = predict_time
        experiment.results.loc[name, 'metric_time'] = metric_time
        experiment.results.loc[name, 'full_time'] = (fit_time + predict_time + metric_time)
        pred.unpersist()

    def build_experiment(self) -> Experiment:
        return Experiment(self.dataset.test, {
            MAP(): self.k,
            NDCG(): self.k,
            HitRate(): self.ks,
            Coverage(self.dataset.log): self.k,
            Surprisal(self.dataset.log): self.k,
            MRR(): self.k
        })

    def build_models(self, algorithms: list[str]) -> dict[str, tuple[Recommender, DataFrame]]:
        # def common_builder(ctor, e):
        #     return ctor(
        #         use_gpu=self.gpu, k=self.k, n_epochs=e,
        #         action_randomization_scale=self.action_randomization_scale,
        #         binarized_relevance=self.binarize_relevance,
        #         negative_examples=not self.take_positive_events,
        #         reward_only_top_k=self.reward_top_k
        #     )

        # if 'cql' in algorithms:
        #     algorithms_and_trains.update({
        #         f'CQL_{e}': (common_builder(CQL, e), cql_train)
        #         for e in self.epochs
        #     })
        #
        # if 'sdac' in algorithms:
        #     algorithms_and_trains.update({
        #         f'SDAC_{e}': (common_builder(SDAC, e), cql_train)
        #         for e in self.epochs
        #     })
        #
        # if 'crr' in algorithms:
        #     algorithms_and_trains.update({
        #         f'CRR_{e}': (common_builder(CRR, e), cql_train)
        #         for e in self.epochs
        #     })

        algorithms = list(map(str.lower, algorithms))
        models = {}
        for alg in algorithms:
            if alg == 'ddpg':
                from replay.models.ddpg import DDPG
                models['DDPG'] = (
                    DDPG(seed=self.seed, user_num=1000, item_num=2500),
                    self.dataset.pos_binary_train
                )
            elif alg == 'als':
                from replay.models import ALSWrap
                models['ALS'] = ALSWrap(seed=self.seed), self.dataset.pos_binary_train
            elif alg == 'knn':
                from replay.models import ItemKNN
                models['KNN'] = ItemKNN(num_neighbours=self.k), self.dataset.pos_binary_train
            elif alg == 'lightfm':
                from replay.models import LightFMWrap
                models['LightFM'] = (
                    LightFMWrap(random_state=self.seed),
                    self.dataset.pos_binary_train
                )
            elif alg == 'ucb':
                from replay.models import UCB
                models['UCB'] = UCB(exploration_coef=0.5), self.dataset.binary_train
            elif alg == 'slim':
                from replay.models import SLIM
                models['SLIM'] = SLIM(seed=self.seed), self.dataset.pos_binary_train
            elif alg == 'rand':
                from replay.models import RandomRec
                models['Rand'] = (
                    RandomRec(seed=self.seed, add_cold=False), self.dataset.pos_binary_train
                )
            elif alg == 'popular':
                from replay.models import RandomRec
                models['Popular'] = (
                    RandomRec(distribution='popular_based', seed=self.seed, add_cold=False),
                    self.dataset.pos_binary_train
                )

        return models


def init_spark_session(
        memory_ratio: float, partitions_to_cpu_ratio: Union[float, int]
) -> SparkSession:
    if isinstance(partitions_to_cpu_ratio, float):
        # CPU ratio
        partitions = int(partitions_to_cpu_ratio * os.cpu_count())
    elif isinstance(partitions_to_cpu_ratio, int):
        # direct int number of partitions
        partitions = partitions_to_cpu_ratio
    else:
        raise ValueError('partitions_to_cpu_ratio should be float or int')

    spark = get_spark_session(
        spark_memory=floor(memory_ratio * psutil.virtual_memory().total / 1024 ** 3),
        shuffle_partitions=partitions,
    )
    spark = State(session=spark).session
    spark.sparkContext.setLogLevel('ERROR')
    return spark


def parse_args():
    def int_or_float(x):
        for f in [int, float]:
            try:
                return f(x)
            except ValueError:
                pass

    parser = ArgumentParser()
    parser.add_argument('--ds', dest='dataset_name', default='MovieLens.100k')
    parser.add_argument('--epochs', dest='epochs', nargs='*', type=int, default=[1])
    parser.add_argument('--part', dest='partitions', type=int_or_float, default=1)
    parser.add_argument('--mem', dest='memory', type=float, default=0.7)
    parser.add_argument('--algos', dest='algorithms', nargs='*', type=str, default=[])
    parser.add_argument('--gpu', dest='gpu', type=int, default=-1)
    parser.add_argument('--seed', dest='seed', type=int, default=1234)

    # experiments
    parser.add_argument('--label', dest='label', default=datetime.datetime.now())
    parser.add_argument('--scale', dest='action_randomization_scale', type=float, default=0.1)
    parser.add_argument('--neg', dest='use_negative_events', action='store_true', default=False)
    parser.add_argument('--rat', dest='rating_based_reward', action='store_true', default=False)
    parser.add_argument('--top', dest='reward_top_k', action='store_true', default=False)

    return parser.parse_args()


def main():
    # shenanigans to turn off countless warnings to clear output
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    warnings.filterwarnings("ignore", category=ExperimentalWarning, append=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning, append=True)

    args = parse_args()
    runner = BareRatingsRunner(
        k=[1, 5, 10], test_ratio=0.2,
        **vars(args)
    )
    runner.run()


if __name__ == '__main__':
    main()

