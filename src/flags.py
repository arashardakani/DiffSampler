import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solving SAT/SMT/Verification problems using neural networks."
    )
    # Problem parameters
    parser.add_argument(
        "--dataset_path",
        "-d",
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
        "--learning_rate",
        "--lr",
        type=float,
        default=10e-1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Total number of gradient descent steps to perform.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--loss_fn",
        type=str,
        default="sigmoid_binary_cross_entropy",
        help="what loss function to use",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="what optimizer to use"
    )

    # experiment util parameters
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether touse cpu",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=-1,
        help="Number of experiments to run",
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
    parser.add_argument(
        "--baseline_name", "-bn",
        type=str,
        default='m22',
        help="Baseline solver names, comma separated",
    )
    parser.add_argument(
        "--baseline_only", "-bo",
        action="store_true",
        help="Whether to run baseline solvers only",
    )
    parser.add_argument(
        "--no_baseline", "-nb",
        action="store_true",
        help="Whether to run baseline solvers",
    )
    # Wandb parameters
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="wandb entity (id) name"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="wandb project name"
    )
    # parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
    parser.add_argument(
        "--wandb_group", type=str, default=None, help="wandb run group name"
    )
    # parser.add_argument(
    #     "--wandb_job_type", type=str, default=None, help="wandb job type descrption"
    # )
    # parser.add_argument(
    #     "--wandb_tags", type=str, default=None, help="wandb tags, comma separated"
    # )

    args = parser.parse_args()

    return args
