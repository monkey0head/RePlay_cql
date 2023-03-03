from replay.models.rl.experiments.run.entrypoint import run_experiment, default_run_arg_parser
from replay.models.rl.experiments.utils.config import LazyTypeResolver


class RunnerRegistry(LazyTypeResolver):
    def resolve(self, name: str):
        if name == 'test.toy_ratings':
            from replay.models.rl.experiments.scenarios.test_toy_ratings import ToyRatingsExperiment
            return ToyRatingsExperiment


if __name__ == "__main__":
    run_experiment(
        arg_parser=default_run_arg_parser(),
        experiment_runner_registry=RunnerRegistry(),
    )
