from run.entrypoint import run_experiment, get_run_command_arg_parser
from scenarios.test_pipeline import TestPipelineExperiment


if __name__ == "__main__":
    run_experiment(
        run_command_parser=get_run_command_arg_parser(),
        experiment_runner_registry={
            'test.pipeline': TestPipelineExperiment,
        }
    )
