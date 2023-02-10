from replay.models.rl.experiments.run.entrypoint import run_experiment, default_run_arg_parser
from replay.models.rl.experiments.scenarios.test_pipeline import TestPipelineExperiment

if __name__ == "__main__":
    run_experiment(
        arg_parser=default_run_arg_parser(),
        experiment_runner_registry={
            'test.pipeline': TestPipelineExperiment,
        }
    )
