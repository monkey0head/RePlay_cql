from typing import Optional

import numpy as np
import pandas as pd
import pyspark.sql.types

from numpy.random import default_rng
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.constants import REC_SCHEMA
from replay.models.base_rec import Recommender


class SoftmaxRec(Recommender):
    can_predict_cold_users = True
    can_predict_cold_items = True
    _search_space = {
        "distribution": {
            "type": "categorical",
            "args": ["popular_based", "relevance", "uniform"],
        },
        "temp": {"type": "uniform", "args": [1e-3, 1e3]},
    }

    item_popularity: DataFrame
    temperature: float
    fill: float

    def __init__(
        self,
        distribution: str = "uniform",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        add_cold: Optional[bool] = True,
    ):
        """
        :param distribution: recommendation strategy:
            "uniform" - all items are sampled uniformly
            "popular_based" - recommend popular items more
        :param temperature: bigger values adjust model towards less popular items
        :param seed: random seed
        :param add_cold: flag to add cold items with minimal probability
        """
        if distribution not in ("popular_based", "relevance", "uniform"):
            raise ValueError(
                "distribution can be one of [popular_based, relevance, uniform]"
            )
        if temperature <= 0 and distribution in {"popular_based", "relevance"}:
            raise ValueError("temperature must be bigger than 0")
        self.distribution = distribution
        self.temperature = temperature
        self.seed = seed
        self.add_cold = add_cold

    @property
    def _init_args(self):
        return {
            "distribution": self.distribution,
            "temperature": self.temperature,
            "seed": self.seed,
            "add_cold": self.add_cold,
        }

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

    def _load_model(self, path: str):
        if self.add_cold:
            fill = self.item_popularity.agg({"probability": "min"}).first()[0]
        else:
            fill = 0
        self.fill = fill

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self.distribution == "popular_based":
            item_popularity = (
                log
                .groupBy("item_idx")
                # find item popularity
                .agg(sf.countDistinct("user_idx").astype("double").alias("popularity"))
            )
            item_popularity = (
                item_popularity
                .crossJoin(
                    item_popularity.select(sf.sum("popularity").alias("sum_popularity"))
                )
                # apply pre-softmax with temperature
                .withColumn(
                    "popularity",
                    # e^( [x - x_max] / temperature )
                    sf.exp(
                        # (sf.col("popularity") - sf.col("max_popularity")) / sf.lit(self.temperature)
                        sf.col("popularity") / sf.col("sum_popularity") / sf.lit(self.temperature)
                    )
                )
            )
        elif self.distribution == "relevance":
            item_popularity = (
                log
                .groupBy("item_idx")
                # item popularity by sum of its relevance
                .agg(sf.sum("relevance").alias("popularity"))
            )
        else:
            item_popularity = (
                log
                .select("item_idx")
                .distinct()
                # constant item popularity
                .withColumn("popularity", sf.lit(1.0))
            )

        self.item_popularity = item_popularity.select(
            sf.col("item_idx"), sf.col("popularity")
        )
        cnt = self.item_popularity.cache().count()
        print(cnt)
        self.fill = (
            self.item_popularity.agg(sf.min("popularity")).first()[0]
            if self.add_cold
            else 0.0
        )

    def _clear_cache(self):
        if hasattr(self, "item_popularity"):
            self.item_popularity.unpersist()

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

        filtered_popularity = self.item_popularity.join(
            items,
            on="item_idx",
            how="right" if self.add_cold else "inner",
        ).fillna(self.fill)

        items_np, probs_np = self._get_ids_and_probs_pd(filtered_popularity)
        print(np.nonzero(probs_np))
        print(probs_np[np.nonzero(probs_np)])
        seed = self.seed

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]
            if seed is not None:
                local_rng = default_rng(seed + user_idx)
            else:
                local_rng = default_rng()
            items_idx = local_rng.choice(
                items_np,
                size=cnt,
                p=probs_np,
                replace=False,
            )
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
                f"LEAST(cnt + {k}, {items_np.shape[0]}) AS cnt",
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )

        return recs
