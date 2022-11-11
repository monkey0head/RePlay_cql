import os
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Process

import wandb
from matplotlib import pyplot as plt

from replay.models.rl.experiments.utils.config import extracted, override_config, TKeyPathValue
from replay.models.rl.experiments.run.argparse import parse_arg
from replay.models.rl.experiments.run.entrypoint import (
    TExperimentRunnerRegistry, resolve_experiment_runner, read_config
)


class Sweep:
    id: str
    project: str
    config: dict
    n_agents: int

    experiment_runner_registry: TExperimentRunnerRegistry

    # sweep runs' shared config
    shared_run_config: dict
    shared_run_config_overrides: list[TKeyPathValue]

    def __init__(
            self, sweep_id: str, config: dict, n_agents: int,
            experiment_runner_registry: TExperimentRunnerRegistry,
            shared_config_overrides: list[TKeyPathValue],
            run_arg_parser: ArgumentParser
    ):
        config, run_command_args, wandb_project = extracted(config, 'command', 'project')
        self.config = config
        self.n_agents = n_agents if n_agents is not None else 1
        self.project = wandb_project
        self.experiment_runner_registry = experiment_runner_registry

        shared_config_filepath = self._extract_agents_shared_config_filepath(
            parser=run_arg_parser, run_command_args=run_command_args
        )
        self.shared_run_config = read_config(shared_config_filepath)
        self.shared_run_config_overrides = shared_config_overrides

        # on Linux machines there's some kind of problem with running sweeps in threads?
        # see https://github.com/wandb/client/issues/1409#issuecomment-870174971
        # and https://github.com/wandb/client/issues/3045#issuecomment-1010435868
        os.environ['WANDB_START_METHOD'] = 'thread'

        if sweep_id is None:
            self.id = wandb.sweep(self.config, project=wandb_project)
        else:
            self.id = sweep_id

    def run(self):
        print(f'==> Sweep {self.id}')

        # TODO: test error handling - we want to terminate [on any error]
        #  a) the whole sweep
        #  b) a single agent
        agent_processes = []
        for _ in range(self.n_agents):
            p = Process(
                target=wandb.agent,
                kwargs={
                    'sweep_id': self.id,
                    'function': self._wandb_agent_entry_point
                }
            )
            p.start()
            agent_processes.append(p)

        for p in agent_processes:
            p.join()

        print(f'<== Sweep {self.id}')

    def _wandb_agent_entry_point(self) -> None:
        # noinspection PyBroadException
        try:
            self._run_provided_config()
        except Exception as _:
            import traceback
            import sys
            # catch it only to print traces to the terminal as wandb doesn't do it in Agents!
            print(traceback.print_exc(), file=sys.stderr)
            # finish explicitly with error code (NB: I tend to think it's not necessary here)
            wandb.finish(1)
            # re-raise after printing so wandb catch it
            raise

    def _run_provided_config(self) -> None:
        # BE CAREFUL: this method is expected to be run in parallel — DO NOT mutate `self` here

        # see comments inside func
        turn_off_gui_for_matplotlib()

        # we know here that it's a sweep-induced run and can expect single sweep run config to be
        # passed via wandb.config, hence we take it and apply all overrides:
        # while concatenating overrides, the order DOES matter: run params, then args
        run = wandb.init()
        wandb.init()
        sweep_overrides = list(map(parse_arg, run.config.items()))
        config_overrides = sweep_overrides + self.shared_run_config_overrides

        # it's important to take COPY of the shared config to prevent mutating `self` state
        config = deepcopy(self.shared_run_config)
        override_config(config, config_overrides)

        # start single run
        runner = resolve_experiment_runner(config, self.experiment_runner_registry)
        runner.run()

    @staticmethod
    def _extract_agents_shared_config_filepath(parser: ArgumentParser, run_command_args):
        # there are several ways to extract config filepath based on different conventions
        # we use parser as the most simplistic and automated,
        # but we could introduce strict positional convention or parse with hands

        args, _ = parser.parse_known_args(run_command_args)
        return args.config_filepath


def turn_off_gui_for_matplotlib():
    # Matplotlib tries to spawn GUI which is prohibited for sub-processes meaning
    # you will encounter kernel core errors. To prevent it we tell matplotlib to
    # not touch GUI at all in each of the spawned sub-processes.
    plt.switch_backend('Agg')
