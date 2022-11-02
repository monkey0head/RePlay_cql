from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rl_experiments.run.runner import Runner
from rl_experiments.utils.config import TConfig

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class TestPipelineExperiment(Runner):
    config: TConfig
    logger: Run | None

    init_time: float
    seed: int

    def __init__(
            self, config: TConfig, seed: int,
            **_
    ):
        super().__init__(config, **config)
        self.init_time = time.time()
        self.print_with_timestamp('==> Init')

        self.seed = seed

    def run(self):
        self.print_with_timestamp('==> Run')
        self.print_with_timestamp('<==')

    def print_with_timestamp(self, text: str):
        print(f'[{time.time() - self.init_time:05.3f}] {text}')
