from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from replay.models.rl.experiments.utils.config import TConfig, GlobalConfig
from replay.models.rl.experiments.utils.timer import timer, print_with_timestamp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class TestPipelineExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int

    top_k: int
    epochs: int

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            top_k: int, epochs: int, dataset: TConfig,
            cuda_device: bool | int | None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=None
        )
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        if cuda_device is not None:
            import torch.cuda
            print(f'CUDA available: {torch.cuda.is_available()}')

        self.seed = seed
        self.top_k = top_k
        self.epochs = epochs

    def run(self):
        self.print_with_timestamp('==> Run')
        self.print_with_timestamp('<==')

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)
