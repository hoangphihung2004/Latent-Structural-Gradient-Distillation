from configs import TrainConfig, parse_train_config
from training.pipeline import DistillationPipeline


def run_training(config: TrainConfig | None = None):
    pipeline = DistillationPipeline(config or TrainConfig())
    return pipeline.run()


def main(argv=None):
    config = parse_train_config(argv)
    return run_training(config)


if __name__ == "__main__":
    main()
