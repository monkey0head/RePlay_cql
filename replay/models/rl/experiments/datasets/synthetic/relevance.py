from __future__ import annotations

import numpy as np


class RelevanceCalculator:
    metric: str
    positive_ratio: float

    def __init__(self, metric: str, positive_ratio: float):
        self.metric = metric
        self.positive_ratio = positive_ratio
        self.relevant_threshold = None

    def similarity(self, users: np.ndarray, items: np.ndarray) -> np.ndarray | float:
        if self.metric == 'l1':
            d = users - items
            return 1 - np.abs(d).mean(axis=-1)
        elif self.metric == 'l2':
            d = users - items
            avg_sq_d = (d ** 2).mean(axis=-1)
            return 1 - np.sqrt(avg_sq_d)
        elif self.metric == 'cosine':
            dot_product = np.sum(users * items, axis=-1)
            users_norm = np.linalg.norm(users, axis=-1)
            items_norm = np.linalg.norm(items, axis=-1)
            return dot_product / (users_norm * items_norm)
        raise ValueError(f'Unknown similarity metric: {self.metric}')

    def calculate(self, users: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        similarity = self.similarity(users, items)
        if self.relevant_threshold is None:
            self.relevant_threshold = np.quantile(
                similarity, 1 - self.positive_ratio, interpolation='lower'
            )
        relevant = similarity >= self.relevant_threshold

        continuous_relevance = similarity
        discrete_relevance = relevant.astype(int)
        return continuous_relevance, discrete_relevance
