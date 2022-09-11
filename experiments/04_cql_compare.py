import datetime
import logging
import os
import time
import warnings
from argparse import ArgumentParser
from math import floor

import pandas as pd
import psutil
import torch
import tqdm
from optuna.exceptions import ExperimentalWarning
from pyspark.sql import functions as sf, SparkSession

from replay.data_preparator import DataPreparator, Indexer
from replay.experiment import Experiment
from replay.metrics import HitRate, NDCG, MAP, MRR, Coverage, Surprisal
from replay.models import ALSWrap, ItemKNN, LightFMWrap, SLIM, UCB, CQL, Wilson, Recommender
from replay.session_handler import State, get_spark_session
from replay.splitters import DateSplitter
from replay.utils import get_log_info


def main():
    start = time.time()

    # shenanigans to turn off countless warnings to clear output
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    warnings.filterwarnings("ignore", category=ExperimentalWarning, append=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning, append=True)

    use_gpu = torch.cuda.is_available()

    args = parse_args()
    ds = args.dataset
    n_epochs: set[int] = set(args.epochs)

    init_spark_session(args.memory, args.partitions)
    df_log, col_mapping = get_dataset(ds)

    print(f'init: \t \t{time.time() - start:.3f}')

    data_preparator = DataPreparator()
    log = data_preparator.transform(columns_mapping=col_mapping, data=df_log)

    K = 10
    K_list_metrics = [1, 5, 10]
    SEED = 12345

    indexer = Indexer()
    indexer.fit(users=log.select('user_id'), items=log.select('item_id'))
    log = indexer.transform(log).cache()

    # train/test split
    date_splitter = DateSplitter(
        test_start=0.2,
        drop_cold_items=True,
        drop_cold_users=True,

    )

    # will consider ratings >= 3 as positive feedback and negative otherwise
    binary_log = log.withColumn(
        'relevance', sf.when(sf.col('relevance') >= 3, sf.lit(1.0)).otherwise(sf.lit(0.))
    ).cache()
    # define split by the binary log
    binary_train, test = date_splitter.split(binary_log)
    test_start = test.agg(sf.min('timestamp')).collect()[0][0]

    pos_binary_train = binary_train.filter(sf.col('relevance') == 1.).cache()
    rew_train = (
        log
        .filter(sf.col('timestamp') < test_start)
        .withColumn(
            'relevance',
            sf
            .when(sf.col('relevance') == sf.lit(1.0), sf.lit(-1.0))
            .when(sf.col('relevance') == sf.lit(2.0), sf.lit(-0.3))
            .when(sf.col('relevance') == sf.lit(3.0), sf.lit(0.25))
            .when(sf.col('relevance') == sf.lit(4.0), sf.lit(0.7))
            .when(sf.col('relevance') == sf.lit(5.0), sf.lit(1.0))
        )
        .cache()
    )

    print('train info:\n', get_log_info(binary_train))
    print('test info:\n', get_log_info(test))
    test_users = test.select('user_idx').distinct().cache()

    # prepare train data for CQL
    if args.binary:
        # binarized relevance
        cql_train = pos_binary_train if args.positive else binary_train
    else:
        # raw relevance transformed into rewards-like
        cql_train = rew_train
        if args.positive:
            cql_train = rew_train.filter(sf.col('relevance') > 0).cache()

    print(f'data prep: \t\t{time.time() - start:.3f}')

    experiment = Experiment(test, {
        MAP(): K, NDCG(): K, HitRate(): K_list_metrics,
        Coverage(log): K, Surprisal(log): K, MRR(): K
    })

    algorithms_and_trains = {
        f'CQL_{e}': (
            CQL(
                use_gpu=use_gpu, k=K, n_epochs=e,
                action_randomization_scale=args.scale,
                binarized_relevance=args.binary,
                negative_examples=not args.positive,
                reward_only_top_k=args.reward_top_k
            ),
            cql_train
        )
        for e in n_epochs
    }

    if not args.cql_only:
        algorithms_and_trains.update({
            'ALS': (ALSWrap(seed=SEED), pos_binary_train),
            'KNN': (ItemKNN(num_neighbours=K), pos_binary_train),
            'LightFM': (LightFMWrap(random_state=SEED), pos_binary_train),
            'UCB': (UCB(exploration_coef=0.5), binary_train)
        })
        if args.test_slim:
            algorithms_and_trains.update({
                'SLIM': (SLIM(seed=SEED), pos_binary_train)
            })

    logger = logging.getLogger("replay")
    results_label = f'{args.label}.{ds}.md'
    print(f'Results are saved to {results_label}')

    for name in tqdm.tqdm(algorithms_and_trains.keys(), desc='Model'):
        model, train = algorithms_and_trains[name]
        logger.info(msg='{} started'.format(name))

        fit_predict_add_res(
            name, model, experiment, train=train, top_k=K, test_users=test_users
        )
        print(
            experiment.results[[
                f'NDCG@{K}', f'MRR@{K}', f'Coverage@{K}', 'fit_time'
            ]].sort_values(f'NDCG@{K}', ascending=False)
        )

        results_md = experiment.results.sort_values(f'NDCG@{K}', ascending=False).to_markdown()
        with open(results_label, 'w') as text_file:
            text_file.write(results_md)

    print(f'finish: \t\t{time.time() - start:.3f}')


def fit_predict_add_res(
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


def init_spark_session(memory_ratio: float, partitions_to_cpu_ratio: float) -> SparkSession:
    spark = get_spark_session(
        spark_memory=floor(memory_ratio * psutil.virtual_memory().total / 1024 ** 3),
        shuffle_partitions=floor(partitions_to_cpu_ratio * os.cpu_count()),
    )
    spark = State(session=spark).session
    spark.sparkContext.setLogLevel('ERROR')
    return spark


def get_dataset(ds_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    ds_name, category = ds_name.split('.')

    if ds_name == 'MovieLens':
        from rs_datasets import MovieLens
        ml = MovieLens(category)
    elif ds_name == 'Amazon':
        from rs_datasets import Amazon
        ml = Amazon(category=category)
    else:
        raise KeyError()

    col_mapping = {
        'user_id': 'user_id',
        'item_id': 'item_id',
        'relevance': 'rating',
        'timestamp': 'timestamp'
    }
    return ml.ratings, col_mapping


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ds', dest='dataset', default='MovieLens.100k')
    parser.add_argument('--epochs', dest='epochs', nargs='*', type=int, default=[1])
    parser.add_argument('--part', dest='partitions', type=float, default=0.8)
    parser.add_argument('--mem', dest='memory', type=float, default=0.7)
    # testing hacks and whistles
    parser.add_argument('--slim', dest='test_slim', action='store_true', default=False)
    parser.add_argument('--cql', dest='cql_only', action='store_true', default=False)
    # parser.add_argument('--cache', dest='cache', action='store_true', default=False)
    parser.add_argument('--label', dest='label', default=datetime.datetime.now())
    # experiments
    parser.add_argument('--scale', dest='scale', type=float, default=0.1)
    parser.add_argument('--pos', dest='positive', action='store_true', default=False)
    parser.add_argument('--bin', dest='binary', action='store_true', default=False)
    parser.add_argument('--top', dest='reward_top_k', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()

