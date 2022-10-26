from typing import Optional

import numpy as np
import pandas as pd
import pyspark.sql.types

from numpy.random import default_rng
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.constants import REC_SCHEMA
from replay.models.base_rec import Recommender


class MostPopularRec(Recommender):
    can_predict_cold_users = True
    can_predict_cold_items = True

    item_popularity: np.ndarray

    def __init__(
        self,
        add_cold: Optional[bool] = True,
    ):
        self.add_cold = add_cold

    @property
    def _init_args(self):
        return {
            "add_cold": self.add_cold,
        }

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.item_popularity = (
            log
            .groupBy("item_idx")
            # find item popularity
            .agg(sf.countDistinct("user_idx").alias("popularity"))
            .orderBy("popularity", ascending=False)
            .select(sf.col("item_idx"))
            .toPandas()['item_idx'].values
        )

    def _get_ids_and_probs_pd(self, item_popularity):
        if self.distribution == "uniform":
            return (
                item_popularity.select("item_idx")
                .toPandas()["item_idx"]
                .values,
                None,
            )

        item_probability = (
            item_popularity
            # normalize popularity to probability
            .crossJoin(
                item_popularity.select(sf.sum("popularity").alias("sum_popularity"))
            )
            .withColumn("probability", sf.col("popularity") / sf.col("sum_popularity"))
        ).toPandas()

        return item_probability["item_idx"].values, item_probability["probability"].values

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        filtered_popularity = self.item_popularity
        if not self.add_cold:
            _items = items.select("item_idx").toPandas()["item_idx"].values
            _, indices, _ = np.intersect1d(
                filtered_popularity, _items, assume_unique=True, return_indices=True
            )
            indices = np.sort(indices)
            filtered_popularity = filtered_popularity[indices]

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]
            items_idx = filtered_popularity[:cnt]
            relevance = 1 / np.arange(1, cnt + 1)
            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_idx,
                    "relevance": relevance,
                }
            )

        recs = (
            log.join(users, how="right", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .agg(sf.countDistinct("item_idx").alias("cnt"))
            .selectExpr(
                "user_idx",
                f"LEAST(cnt + {k}, {filtered_popularity.shape[0]}) AS cnt",
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )

        return recs
