# pylint: skip-file

import numpy as np
import pandas as pd
import pytest

import math
import pyspark.sql.functions as sf

from replay.metrics import *
from replay.distributions import item_distribution
from replay.metrics.base_metric import sorter

from tests.utils import *


@pytest.fixture
def one_user():
    df = pd.DataFrame({"user_idx": [1], "item_idx": [1], "relevance": [1]})
    return df


@pytest.fixture
def two_users():
    df = pd.DataFrame(
        {"user_idx": [1, 2], "item_idx": [1, 2], "relevance": [1, 1]}
    )
    return df


@pytest.fixture
def recs(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 3.0],
            [0, 1, 2.0],
            [0, 2, 1.0],
            [1, 0, 3.0],
            [1, 1, 4.0],
            [1, 4, 1.0],
            [2, 0, 5.0],
            [2, 2, 1.0],
            [2, 3, 2.0],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def recs2(spark):
    return spark.createDataFrame(
        data=[[0, 3, 4.0], [0, 4, 5.0]],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def empty_recs(spark):
    return spark.createDataFrame(
        data=[],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def true(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 9, 12), 3.0],
            [0, 4, datetime(2019, 9, 13), 2.0],
            [0, 1, datetime(2019, 9, 17), 1.0],
            [1, 5, datetime(2019, 9, 14), 4.0],
            [1, 0, datetime(2019, 9, 15), 3.0],
            [2, 1, datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def prev_relevance(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 100.0],
            [0, 4, 0.0],
            [1, 10, -5.0],
            [4, 6, 11.5],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def quality_metrics():
    return [NDCG(), HitRate(), Precision(), Recall(), MAP(), MRR(), RocAuc()]


@pytest.fixture
def duplicate_recs(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 3.0],
            [0, 1, 2.0],
            [0, 2, 1.0],
            [0, 0, 3.0],
            [1, 0, 3.0],
            [1, 1, 4.0],
            [1, 4, 1.0],
            [1, 1, 2.0],
            [2, 0, 5.0],
            [2, 2, 1.0],
            [2, 3, 2.0],
        ],
        schema=REC_SCHEMA,
    )


def test_test_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(one_user, two_users, 1) == 0.5, str(metric)


def test_pred_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(two_users, one_user, 1) == 1.0, str(metric)


def test_hit_rate_at_k(recs, true):
    assertDictAlmostEqual(
        HitRate()(recs, true, [3, 1]),
        {3: 2 / 3, 1: 1 / 3},
    )


def test_user_dist(log, recs, true):
    vals = HitRate().user_distribution(log, recs, true, 1)["value"].to_list()
    assert_allclose(vals, [0.0, 0.5])


def test_item_dist(log, recs):
    assert_allclose(
        item_distribution(log, recs, 1)["rec_count"].to_list(),
        [0, 0, 1, 2],
    )


def test_ndcg_at_k(recs, true):
    assertDictAlmostEqual(
        NDCG()(recs, true, [1, 3]),
        {
            1: 1 / 3,
            3: 1
            / 3
            * (
                1
                / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))
                * (1 / np.log2(2) + 1 / np.log2(3))
                + 1 / (1 / np.log2(2) + 1 / np.log2(3)) * (1 / np.log2(3))
            ),
        },
    )


def test_precision_at_k(recs, true):
    assertDictAlmostEqual(
        Precision()(recs, true, [1, 2, 3]),
        {3: 1 / 3, 1: 1 / 3, 2: 1 / 2},
    )


def test_map_at_k(recs, true):
    assertDictAlmostEqual(
        MAP()(recs, true, [1, 3]),
        {3: 11 / 36, 1: 1 / 3},
    )


def test_recall_at_k(recs, true):
    assertDictAlmostEqual(
        Recall()(recs, true, [1, 3]),
        {3: (1 / 2 + 2 / 3) / 3, 1: 1 / 9},
    )


def test_surprisal_at_k(true, recs, recs2):
    assertDictAlmostEqual(Surprisal(true)(recs2, [1, 2]), {1: 1.0, 2: 1.0})

    assert_allclose(
        Surprisal(true)(recs, 3),
        5 * (1 - 1 / np.log2(3)) / 9 + 4 / 9,
    )


def test_unexpectedness_at_k(true, recs, recs2):
    assert Unexpectedness._get_metric_value_by_user(2, (), (2, 3)) == 0
    assert Unexpectedness._get_metric_value_by_user(2, (1, 2), (1,)) == 0.5


def test_coverage(true, recs, empty_recs):
    coverage = Coverage(recs.union(true.drop("timestamp")))
    assertDictAlmostEqual(
        coverage(recs, [1, 3, 5]),
        {1: 0.3333333333333333, 3: 0.8333333333333334, 5: 0.8333333333333334},
    )
    assertDictAlmostEqual(
        coverage(empty_recs, [1, 3, 5]),
        {1: 0.0, 3: 0.0, 5: 0.0},
    )


def test_bad_coverage(true, recs):
    assert_allclose(Coverage(true)(recs, 3), 1.25)


def test_empty_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[], ground_truth=[2, 4]
            ),
            0,
            err_msg=str(metric),
        )


def test_bad_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[1, 3], ground_truth=[2, 4]
            ),
            0,
            err_msg=str(metric),
        )


def test_not_full_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[4, 1, 2], ground_truth=[2, 4]
            ),
            metric._get_metric_value_by_user(
                k=3, pred=[4, 1, 2], ground_truth=[2, 4]
            ),
            err_msg=str(metric),
        )


def test_duplicate_recs(quality_metrics, duplicate_recs, recs, true):
    for metric in quality_metrics:
        assert_allclose(
            metric(k=4, recommendations=duplicate_recs, ground_truth=true),
            metric(k=4, recommendations=recs, ground_truth=true),
            err_msg=str(metric),
        )


def test_sorter():
    result = sorter(((1, 2), (2, 3), (3, 2)))
    assert result == [2, 3]


def test_sorter_index():
    ids, weights = sorter(((1, 2, 3), (2, 3, 4), (3, 3, 5)), extra_position=2)
    assert ids == [3, 2]
    assert weights == [5, 3]


def test_ncis_raises(recs, prev_relevance, true):
    with pytest.raises(ValueError):
        NCISPrecision(prev_policy_weights=prev_relevance, activation="absent")


def test_ncis_activations_softmax(spark, prev_relevance):
    res = NCISPrecision._softmax_by_user(prev_relevance, "relevance")
    gt = spark.createDataFrame(
        data=[
            [0, 0, math.e**100 / (math.e**100 + math.e**0)],
            [0, 4, math.e**0 / (math.e**100 + math.e**0)],
            [1, 10, math.e**0 / (math.e**0)],
            [4, 6, math.e**0 / (math.e**0)],
        ],
        schema=REC_SCHEMA,
    )
    sparkDataFrameEqual(res, gt)


def test_ncis_activations_sigmoid(spark, prev_relevance):
    res = NCISPrecision._sigmoid(prev_relevance, "relevance")
    gt = spark.createDataFrame(
        data=[
            [0, 0, 1 / (1 + math.e ** (-100))],
            [0, 4, 1 / (1 + math.e**0)],
            [1, 10, 1 / (1 + math.e**5)],
            [4, 6, 1 / (1 + math.e ** (-11.5))],
        ],
        schema=REC_SCHEMA,
    )
    sparkDataFrameEqual(res, gt)


def test_ncis_weigh_and_clip(spark, prev_relevance):
    res = NCISPrecision._weigh_and_clip(
        df=(
            prev_relevance.withColumn(
                "prev_relevance",
                sf.when(sf.col("user_idx") == 1, sf.lit(0)).otherwise(
                    sf.lit(20)
                ),
            )
        ),
        threshold=10,
    )
    gt = spark.createDataFrame(
        data=[[0, 0, 5.0], [0, 4, 0.1], [1, 10, 10.0], [4, 6, 11.5 / 20]],
        schema=REC_SCHEMA,
    ).withColumnRenamed("relevance", "weight")
    sparkDataFrameEqual(res.select("user_idx", "item_idx", "weight"), gt)


def test_ncis_get_enriched_recommendations(spark, recs, prev_relevance, true):
    ncis_precision = NCISPrecision(prev_policy_weights=prev_relevance)
    enriched = ncis_precision._get_enriched_recommendations(recs, true, 3)
    gt = spark.createDataFrame(
        data=[
            [0, ([0, 1, 2]), ([0.1, 10.0, 10.0]), ([0, 1, 4])],
            [1, ([1, 0, 4]), ([10.0, 10.0, 10.0]), ([0, 5])],
            [2, ([0, 3, 2]), ([10.0, 10.0, 10.0]), ([1])],
        ],
        schema="user_idx int, pred array<int>, weight array<double>, ground_truth array<int>",
    ).withColumnRenamed("relevance", "weight")
    sparkDataFrameEqual(enriched, gt)


def test_ncis_precision(recs, prev_relevance, true):
    ncis_precision = NCISPrecision(prev_policy_weights=prev_relevance)
    assert (
        ncis_precision._get_metric_value_by_user(
            4, [1, 0, 4], [0, 5, 4], [20.0, 5.0, 15.0]
        )
        == 0.5
    )
    assert ncis_precision._get_metric_value_by_user(4, [], [0, 5, 4], []) == 0
    assert (
        ncis_precision._get_metric_value_by_user(4, [1], [0, 5, 4], [100]) == 0
    )
    assert (
        ncis_precision._get_metric_value_by_user(4, [1], [1, 5, 4], [100]) == 1
    )


def test_precision(recs, true):
    precision = Precision()
    assert precision._get_metric_value_by_user(4, [1], [1, 5, 4]) == 1
