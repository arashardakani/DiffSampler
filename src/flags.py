import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solving SAT/SMT/Verification problems using neural networks."
    )
    # Problem parameters
    parser.add_argument(
        "--dataset_path","-d",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--use_pgates", type=bool, default=True, help="Whether to use pgates or not"
    )

    # train(backprop) parameters
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        default=10e-1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1, help="Batch size per GPU/CPU for training."
    )
    
    parser.add_argument(
        "--loss_fn", type=str, default="sigmoid_binary_cross_entropy", help="what mode of probability modeling to use"
    )

    # util parameters
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether touse cpu",
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
