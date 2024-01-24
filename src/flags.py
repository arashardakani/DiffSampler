import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solving SAT/SMT/Verification problems using neural networks."
    )
    # Problem parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=45, help="random seed for initialization"
    )
    parser.add_argument(
        "--use_pgates", type=bool, default=True, help="Whether to use pgates or not"
    )

    # train(backprop) parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=100e0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="Top k predictions to consider"
    )
    parser.add_argument(
        "--loss_fn", type=str, default="mse", help="Which loss function to use"
    )
    parser.add_argument(
        "--early_exit", "-e", action="store_true", help="Whether to use early exit"
    )

    # util parameters
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print training logs or not",
    )
    parser.add_argument(
        "--latency_experiment",
        "-l",
        action="store_true",
        help="Whether to run latency experiment or not",
    )
    parser.add_argument(
        "--dump_solution",
        action="store_true",
        help="Whether to dump solution or not",
    )
    args = parser.parse_args()

    return args
