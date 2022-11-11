from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch.cuda

from replay.models.rl.experiments.run.runner import Runner
from replay.models.rl.experiments.utils.config import TConfig
from replay.models.rl.experiments.utils.rating_dataset import RatingDataset

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class TestPipelineExperiment(Runner):
    config: TConfig
    logger: Run | None

    init_time: float
    seed: int

    k: int
    epochs: int
    dataset: RatingDataset

    def __init__(
            self, config: TConfig, seed: int,
            k: int, epochs: int, dataset: TConfig,
            **_
    ):
        super().__init__(config, **config)
        self.init_time = time.time()
        self.print_with_timestamp('==> Init')
        print(f'CUDA available: {torch.cuda.is_available()}')

        self.seed = seed
        self.k = k
        self.epochs = epochs
        self.dataset = RatingDataset(k=k, **dataset)

    def run(self):
        self.print_with_timestamp('==> Run')

        (
            train_dataset, user_logs_train, test_dataset, users_logs_test
        ) = self.dataset.prepare()

        from replay.models.rl.sdac.sdac import SDAC
        from replay.models.rl.experiments.utils.encoders import CustomEncoderFactory
        from replay.models.rl.experiments.utils.fake_recommender_env import FakeRecomenderEnv
        from d3rlpy.metrics import evaluate_on_environment

        sdac = SDAC(
            use_gpu=False,
            actor_encoder_factory=CustomEncoderFactory(64),
            critic_encoder_factory=CustomEncoderFactory(64),
            encoder_factory=CustomEncoderFactory(64)
        )
        env = FakeRecomenderEnv(
            logger=self.logger, test_data=users_logs_test[:10000], top_k=self.k
        )
        evaluate_scorer = evaluate_on_environment(env)
        sdac.fit(
            train_dataset,
            eval_episodes=train_dataset,
            n_epochs=self.epochs,
            scorers={'environment': evaluate_scorer}
        )

        self.print_with_timestamp('<==')

    def print_with_timestamp(self, text: str):
        print(f'[{time.time() - self.init_time:5.1f}] {text}')
