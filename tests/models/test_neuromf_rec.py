"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
import os
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.models.neuromf import NMF, NeuroMF


class NeuroCFRecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {
            "learning_rate": 0.5,
            "epochs": 1,
            "embedding_gmf_dim": 2,
            "embedding_mlp_dim": 2,
            "hidden_mlp_dims": [2],
        }
        self.model = NeuroMF(**params)
        self.log = self.spark.createDataFrame(
            [
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "1", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "1", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "1", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
            ],
            schema=LOG_SCHEMA,
        )

    def test_predict(self):
        self.model.fit(log=self.log)
        predictions = self.model.predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            filter_seen_items=True,
        )
        self.assertTrue(
            np.allclose(
                predictions.toPandas()[["user_id", "item_id"]]
                .astype(int)
                .values,
                [[0, 0], [1, 2]],
                atol=1.0e-3,
            )
        )

    def test_check_gmf_only(self):
        params = {"learning_rate": 0.5, "epochs": 1, "embedding_gmf_dim": 2}
        raised = False
        self.model = NeuroMF(**params)
        try:
            self.model.fit(log=self.log)
        except RuntimeError:
            raised = True
        self.assertFalse(raised)

    def test_check_mlp_only(self):
        params = {
            "learning_rate": 0.5,
            "epochs": 1,
            "embedding_mlp_dim": 2,
            "hidden_mlp_dims": [2],
        }
        raised = False
        self.model = NeuroMF(**params)
        try:
            self.model.fit(log=self.log)
        except RuntimeError:
            raised = True
        self.assertFalse(raised)

    def test_check_simple_mlp_only(self):
        params = {"learning_rate": 0.5, "epochs": 1, "embedding_mlp_dim": 2}
        raised = False
        self.model = NeuroMF(**params)
        try:
            self.model.fit(log=self.log)
        except RuntimeError:
            raised = True
        self.assertFalse(raised)

    def test_empty_embeddings_exception(self):
        self.assertRaises(
            ValueError, NeuroMF,
        )

    def test_negative_dims_exception(self):
        self.assertRaises(
            ValueError, NeuroMF, embedding_gmf_dim=-2, embedding_mlp_dim=-1,
        )
