import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Type

from ruamel import yaml

from rl_experiments.run.argparse import parse_arg
from rl_experiments.utils.config import (
    extracted_type, TConfig, override_config
)
from rl_experiments.run.runner import Runner

TExperimentRunnerRegistry = dict[str, Type[Runner]]


def run_experiment(
        run_command_parser: ArgumentParser,
        experiment_runner_registry: TExperimentRunnerRegistry
) -> None:
    args, unknown_args = run_command_parser.parse_known_args()

    config = read_config(args.config_filepath)
    config_overrides = list(map(parse_arg, unknown_args))

    if args.wandb_entity:
        # overwrite wandb entity for the run
        os.environ['WANDB_ENTITY'] = args.wandb_entity

    # prevent math parallelization as it usually only slows things down for us
    set_single_threaded_math()

    if args.wandb_sweep:
        from rl_experiments.run.sweep import Sweep
        Sweep(
            sweep_id=args.wandb_sweep_id,
            config=config,
            n_agents=args.n_sweep_agents,
            experiment_runner_registry=experiment_runner_registry,
            shared_config_overrides=config_overrides,
            run_arg_parser=run_command_parser,
        ).run()
    else:
        override_config(config, config_overrides)
        runner = resolve_experiment_runner(config, experiment_runner_registry)
        runner.run()


def set_single_threaded_math():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def get_run_command_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    # todo: add examples
    # todo: remove --sweep ?
    parser.add_argument('-c', '--config', dest='config_filepath', required=True)
    parser.add_argument('-e', '--entity', dest='wandb_entity', required=False, default=None)
    parser.add_argument('--sweep', dest='wandb_sweep', action='store_true', default=False)
    parser.add_argument('--sweep_id', dest='wandb_sweep_id', default=None)
    parser.add_argument('-n', '--n_sweep_agents', type=int, default=None)
    return parser


def resolve_experiment_runner(
        config: TConfig,
        experiment_runner_registry: TExperimentRunnerRegistry
) -> Runner:
    config, experiment_type = extracted_type(config)
    runner_cls = experiment_runner_registry.get(experiment_type, None)

    assert runner_cls, f'Experiment runner type "{experiment_type}" is not supported'
    return runner_cls(config, **config)


def read_config(filepath: str) -> TConfig:
    filepath = Path(filepath)
    with filepath.open('r') as config_io:
        return yaml.load(config_io, Loader=yaml.Loader)
