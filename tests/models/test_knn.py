# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.models import KNN
from tests.utils import spark


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [1, 1, date, 1.0],
            [2, 0, date, 1.0],
            [2, 1, date, 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def weighting_log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [0, 1, date, 1.0],
            [1, 1, date, 1.0],
            [2, 0, date, 1.0],
            [2, 1, date, 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    model = KNN(1, weighting=None)
    return model


@pytest.fixture
def tf_idf_model():
    model = KNN(1, weighting="tf_idf")
    return model


@pytest.fixture
def bm25_model():
    model = KNN(1, weighting="bm25")
    return model


def test_works(log, model):
    model.fit(log)
    recs = model.predict(log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 1
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


def test_tf_idf(weighting_log, tf_idf_model):
    idf1, idf2 = tf_idf_model._get_idf(weighting_log)
    idf1 = idf1.toPandas()
    idf2 = idf2.toPandas()
    assert np.array_equal(idf1.values, idf2.values)
    assert np.allclose(
        idf1[idf1["item_idx_one"] == 0]["idf1"], np.log1p(3 / 2)
    )
    assert np.allclose(
        idf1[idf1["item_idx_one"] == 1]["idf1"], np.log1p(3 / 3)
    )

    tf_idf_model.fit(weighting_log)
    recs = tf_idf_model.predict(weighting_log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


def test_bm25(weighting_log, bm25_model):
    k1 = bm25_model.k1
    b = bm25_model.b
    avgdl = (1 + 2 + 2) / 3

    tf1, tf2 = bm25_model._get_tf_bm25(weighting_log)
    tf1 = tf1.toPandas()
    tf2 = tf2.toPandas()
    assert np.array_equal(tf1.values, tf2.values)
    assert np.allclose(
        tf1[tf1["user_idx"] == 0]["rel_one"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / avgdl)),
    )
    assert np.allclose(
        tf1[tf1["user_idx"] == 1]["rel_one"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 1 / avgdl)),
    )
    assert np.allclose(
        tf1[tf1["user_idx"] == 2]["rel_one"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / avgdl)),
    )

    idf1, idf2 = bm25_model._get_idf_bm25(weighting_log)
    idf1 = idf1.toPandas()
    idf2 = idf2.toPandas()
    assert np.array_equal(idf1.values, idf2.values)
    assert np.allclose(
        idf1[idf1["item_idx_one"] == 0]["idf1"],
        np.log1p((3 - 2 + 0.5) / (2 + 0.5)),
    )
    assert np.allclose(
        idf1[idf1["item_idx_one"] == 1]["idf1"],
        np.log1p((3 - 3 + 0.5) / (3 + 0.5)),
    )

    bm25_model.fit(weighting_log)
    recs = bm25_model.predict(weighting_log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0
