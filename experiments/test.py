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
from pyspark.sql import functions as sf

from replay.data_preparator import DataPreparator, Indexer
from replay.experiment import Experiment
from replay.metrics import HitRate, NDCG, MAP, MRR, Coverage, Surprisal
from replay.models import UCB, CQL, Wilson, Recommender
from replay.session_handler import State, get_spark_session
from replay.splitters import DateSplitter
from replay.utils import get_log_info


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

    pred = model.predict(log=train, k=top_k, users=test_users)
    pred.cache()
    pred.count()
    predict_time = time.time() - start_time - fit_time

    experiment.add_result(name, pred)
    metric_time = time.time() - start_time - fit_time - predict_time

    experiment.results.loc[name, 'fit_time'] = fit_time
    experiment.results.loc[name, 'predict_time'] = predict_time
    experiment.results.loc[name, 'metric_time'] = metric_time
    experiment.results.loc[name, 'full_time'] = (fit_time + predict_time + metric_time)
    pred.unpersist()


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


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    use_gpu = torch.cuda.is_available()

    parser = ArgumentParser()
    parser.add_argument('--ds', dest='dataset', required=False, default='MovieLens.100k')
    parser.add_argument('--epochs', dest='epochs', nargs='*', type=int, required=True)
    parser.add_argument('--part', dest='partitions', type=float, required=False, default=0.8)
    parser.add_argument('--mem', dest='memory', type=float, required=False, default=0.7)
    parser.add_argument(
        '--scale', dest='action_randomization_scale', type=float, required=False, default=0.01
    )

    args = parser.parse_args()
    ds = args.dataset
    n_epochs: set[int] = set(args.epochs)

    spark = get_spark_session(
        spark_memory=floor(psutil.virtual_memory().total / 1024 ** 3 * args.memory),
        shuffle_partitions=floor(os.cpu_count() * args.partitions),
    )
    spark = State(session=spark).session
    spark.sparkContext.setLogLevel('ERROR')

    K = 10
    K_list_metrics = [1, 5, 10]

    if os.path.exists('./train.parquet'):
        train = spark.read.parquet('./train.parquet')
        test = spark.read.parquet('./test.parquet')
    else:
        df_log, col_mapping = get_dataset(ds)

        data_preparator = DataPreparator()
        log = data_preparator.transform(columns_mapping=col_mapping, data=df_log)

        indexer = Indexer()
        indexer.fit(users=log.select('user_id'), items=log.select('item_id'))

        # will consider ratings >= 3 as positive feedback.
        # A positive feedback is treated with relevance = 1
        only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1.))
        # negative feedback will be used for Wilson and UCB models
        only_negatives_log = log.filter(sf.col('relevance') < 3).withColumn('relevance', sf.lit(0.))

        pos_log = indexer.transform(df=only_positives_log)

        # train/test split
        date_splitter = DateSplitter(
            test_start=0.2,
            drop_cold_items=True,
            drop_cold_users=True,

        )
        train, test = date_splitter.split(pos_log)
        train.cache(), test.cache()
        train.write.parquet('./train.parquet')
        test.write.parquet('./test.parquet')
        print('train info:\n', get_log_info(train))
        print('test info:\n', get_log_info(test))

        test_start = test.agg(sf.min('timestamp')).collect()[0][0]

        # train with both positive and negative feedback
        pos_neg_train = (
            train
            .withColumn('relevance', sf.lit(1.))
            .union(
                indexer.transform(
                    only_negatives_log.filter(sf.col('timestamp') < test_start)
                )
            )
        )
        pos_neg_train.cache()
        pos_neg_train.count()

    experiment = Experiment(test, {
        MAP(): K, NDCG(): K, HitRate(): K_list_metrics, Coverage(train): K, Surprisal(train): K,
        MRR(): K
    })

    algorithms = {
        f'CQL_{e}': CQL(
            use_gpu=use_gpu, top_k=K, n_epochs=e,
            action_randomization_scale=args.action_randomization_scale,
            batch_size=2048
        )
        for e in n_epochs
    }
    logger = logging.getLogger("replay")
    test_users = test.select('user_idx').distinct()

    for name in tqdm.tqdm(algorithms.keys(), desc='Model'):
        model = algorithms[name]

        logger.info(msg='{} started'.format(name))

        train_ = train
        fit_predict_add_res(name, model, experiment, train=train_, top_k=K, test_users=test_users)
        print(
            experiment.results[[
                f'NDCG@{K}', f'MRR@{K}', f'Coverage@{K}', 'fit_time'
            ]].sort_values(f'NDCG@{K}', ascending=False)
        )

        results_md = experiment.results.sort_values(f'NDCG@{K}', ascending=False).to_markdown()
        with open(f'{ds}.md', 'w') as text_file:
            text_file.write(results_md)


if __name__ == '__main__':
    main()
