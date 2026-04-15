import argparse

from .defaults import TrainConfig


def build_train_arg_parser() -> argparse.ArgumentParser:
    default = TrainConfig()

    parser = argparse.ArgumentParser(
        description="Train latent structural gradient distillation models.",
    )
    parser.add_argument("--dataset-name", type=str, default=default.dataset_name)
    parser.add_argument("--batch-size", type=int, default=default.batch_size)
    parser.add_argument("--learning-rate", type=float, default=default.learning_rate)
    parser.add_argument("--epochs", type=int, default=default.epochs)
    parser.add_argument("--alpha", type=float, default=default.alpha)
    parser.add_argument("--delta", type=float, default=default.delta)
    parser.add_argument("--num-workers", type=int, default=default.num_workers)
    parser.add_argument("--patience", type=int, default=default.patience)
    parser.add_argument("--data-root", type=str, default=default.data_root)
    parser.add_argument("--output-dir", type=str, default=default.output_dir)

    parser.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="Enable early stopping.",
    )
    parser.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable early stopping.",
    )
    parser.set_defaults(early_stop=default.early_stop)

    return parser


def parse_train_config(argv=None) -> TrainConfig:
    parser = build_train_arg_parser()
    args = parser.parse_args(argv)

    return TrainConfig(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        alpha=args.alpha,
        delta=args.delta,
        num_workers=args.num_workers,
        early_stop=args.early_stop,
        patience=args.patience,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
