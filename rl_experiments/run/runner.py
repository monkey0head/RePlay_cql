from __future__ import annotations

from typing import TYPE_CHECKING

from rl_experiments.utils.config import TConfig

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class Runner:
    config: TConfig
    logger: Run | None

    # noinspection PyUnusedLocal
    def __init__(
            self, config: TConfig, log: bool = False, project: str = None,
            **unpacked_config
    ):
        self.config = config
        self.logger = None
        if log:
            import wandb
            self.logger = wandb.init(project=project)
            # we have to pass the config with update instead of init because of sweep runs
            self.logger.config.update(self.config)

    def run(self) -> None:
        ...
