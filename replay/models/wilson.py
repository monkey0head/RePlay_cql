"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Optional

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from statsmodels.stats.proportion import proportion_confint

from replay.utils import convert2spark
from replay.models.pop_rec import PopRec


class Wilson(PopRec):
    """
    Подсчитевает для каждого айтема нижнюю границу
    доверительного интервала истинной доли положительных оценок.

    Для каждого пользователя отфильтровываются просмотренные айтемы.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2], "relevance": [1, 1]})
    >>> model = Wilson()
    >>> model.fit_predict(data_frame,k=1).toPandas()
      user_id item_id  relevance
    0       1       2   0.206549
    1       2       1   0.206549

    """

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        data_frame = (
            log.groupby("item_idx")
            .agg(
                sf.sum("relevance").alias("pos"),
                sf.count("relevance").alias("total"),
            )
            .toPandas()
        )
        pos = np.array(data_frame["pos"].values)
        total = np.array(data_frame["total"].values)
        data_frame["relevance"] = proportion_confint(
            pos, total, method="wilson"
        )[0]
        data_frame = data_frame.drop(["pos", "total"], axis=1)
        self.item_popularity = convert2spark(data_frame).cache()
