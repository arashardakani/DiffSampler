import glob
import logging
import os
import pathlib
import random
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
from pysat.solvers import Solver
from tqdm import tqdm
import wandb

import flags

# import model.circuit_jax as circuit
from model.initlialize_sat_prob import init_problem
from model.gdsolve import gdsolve
from model.gdsolve_verbose import gdsolve_verbose

from utils.baseline_sat import BaselineSolverRunner

logging.basicConfig(level=logging.INFO)


class Runner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        if not self.args.latency_experiment:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Args: {self.problem_type}")
            logging.info(
                "\n".join([f"{k}: {v}" for k, v in self.args.__dict__.items()])
            )

        # set random seed
        random.seed(self.args.seed)
        self.key = jax.random.PRNGKey(self.args.seed)

        # setup the dataset
        self.datasets = []
        self.dataset_str = ""
        self._setup_problems()

        # setup the save directory
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.save_dir = self.save_dir / (
            f"{self.dataset_str}_{self.args.batch_size}_{self.args.optimizer}_{self.args.learning_rate}_{datetime.now().strftime('%Y%m%d')}"
            + "_latency"
            if self.args.latency_experiment
            else "_logging"
        )
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # use CPU if specified
        if self.args.use_cpu:
            jax.config.update("jax_platform_name", "cpu")

        # setup wandb
        self.do_wandb = (
            self.args.wandb_entity is not None
            and not self.args.latency_experiment
            and not self.args.debug
        )
        if self.do_wandb:
            logging.info("Logging to wandb")
            assert self.args.wandb_project is not None
            self.wandb_config = {
                "batch_size": self.args.batch_size,
                "lr": self.args.learning_rate,
                "steps": self.args.num_steps,
                "train_data_path": self.args.dataset_path,
                "optimizer": self.args.optimizer,
            }

    def _setup_problems(self):
        """Setup the problem.
        Implementation will later be extended to support other problem types. (currnetly only SAT)
            - SAT: dataset will be either (1) .cnf files in args.dataset_path or (2) PySAT PHP problems.
        """
        if self.problem_type == "sat":
            assert self.args.dataset_path is not None
            dataset_path = os.path.join(
                pathlib.Path(__file__).parent.parent, self.args.dataset_path
            )
            self.datasets = sorted(glob.glob(dataset_path), key=os.path.getsize)
            if self.args.num_experiments > 0:
                self.datasets = self.datasets[: self.args.num_experiments]
            self.dataset_str = self.args.dataset_path.split("/")[1]
            self.results = {}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError

    def run_model(self, problem: CNF, prob_id: int = 0):
        # wandb logging
        prob_name = self.datasets[prob_id].split("/")[-1].split(".cnf")[0]
        if self.do_wandb:
            wandb_init_config = {
                "project": self.args.wandb_project,
                "entity": self.args.wandb_entity,
                "name": prob_name,
                "id": prob_name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "resume": "allow",
                "group": self.args.wandb_group
                + f"_{self.args.batch_size}_{self.args.optimizer}_{self.args.learning_rate}_"
                + "_".join(self.args.wandb_tags.split(",")),
                "tags": self.args.wandb_tags.split(","),
                "config": self.wandb_config,
            }
        else:
            wandb_init_config = {}
        solutions_found = []
        params, optimizer, literal_tensor, gradient_mask = init_problem(
            cnf_problem=problem,
            batch_size=self.args.batch_size,
            key=self.key,
            learning_rate=self.args.learning_rate,
            optimizer_str=self.args.optimizer,
        )
        log_dict = {}
        result_dict = {}
        if self.args.latency_experiment:
            (
                params,
                steps_ran,
                loss,
                elapsed_time,
                solutions_found,
            ) = gdsolve(
                num_steps=self.args.num_steps,
                params=params,
                optimizer=optimizer,
                literal_tensor=literal_tensor,
                gradient_mask=gradient_mask,
            )
        else:
            (
                params,
                steps_ran,
                loss,
                elapsed_time,
                solutions_found,
                verbose_log_dict,
                all_params,
                all_losses,
            ) = gdsolve_verbose(
                num_steps=self.args.num_steps,
                params=params,
                optimizer=optimizer,
                literal_tensor=literal_tensor,
                gradient_mask=gradient_mask,
                do_wandb=self.do_wandb,
                wandb_init_config=wandb_init_config,
            )
            log_dict = verbose_log_dict
            elapsed_time = 0
            self.export_vector(all_params, prob_id, tag="emb")
            self.export_vector(all_losses, prob_id, tag="loss")
        logging.info("--------------------")
        logging.info("Differential model solving")
        logging.critical(
            f"Elapsed Time: {elapsed_time:.6f} seconds" f" Ran for {steps_ran} steps"
        )
        is_verified = len(solutions_found) > 0
        if is_verified:
            logging.critical("Model solution verified")
        else:
            logging.critical("No solution")
        result_dict.update(
            {
                "model_runtime": elapsed_time,
                "model_steps_ran": steps_ran,
                "model_result": is_verified,
            }
        )
        if self.args.dump_solution:
            self.results[prob_id].update(
                {"model_solution": self.model.get_input_weights(solutions_found[i])}
            )

        return log_dict, result_dict

    def run_baseline(self, problem: CNF, prob_id: int = 0):
        """Run the baseline solver.

        Args:
            problem: CNF problem to be solved
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        solver_names = self.args.baseline_name.split(",")
        results = {}
        for solver_name in solver_names:
            baseline = BaselineSolverRunner(
                cnf_problem=problem, solver_name=solver_name
            )
            baseline_elapsed_time, result, solver_solutions = baseline.run()
            logging.critical(
                f"Baseline {solver_name} Elapsed Time: {baseline_elapsed_time:.6f} seconds."
            )
            results.update(
                {
                    f"baseline_{solver_name}_runtime": baseline_elapsed_time,
                    f"baseline_{solver_name}_result": result,
                }
            )
            if self.args.dump_solution:
                results.update({f"baseline_{solver_name}_solution": solver_solutions})
        return results

    def load_problem(self, prob_id: int):
        """Load a problem from the dataset.

        Args:
            prob_id (int): Index of the problem to be loaded.
        """
        logging.info(f"Loading problem from {self.datasets[prob_id]}")
        problem = CNF(from_file=self.datasets[prob_id])
        logging_dict = {
            "prob_desc": self.datasets[prob_id].split("/")[-1],
            "num_vars": problem.nv,
            "num_clauses": len(problem.clauses),
        }
        logging.info(f"num_vars: {problem.nv}")
        logging.info(f"num_clauses: {len(problem.clauses)}")
        return problem, logging_dict

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        self.problem_name = self.datasets[prob_id].split("/")[-1].split(".cnf")[0]
        pathlib.Path(self.save_dir / self.problem_name).mkdir(
            parents=True, exist_ok=True
        )
        problem, results_dict = self.load_problem(prob_id)
        results = {}
        if not self.args.baseline_only:
            run_logs, model_results = self.run_model(problem, prob_id)
            results_dict.update(model_results)
        if not self.args.no_baseline:
            baseline_results = self.run_baseline(problem, prob_id)
            results_dict.update(baseline_results)
        if self.args.latency_experiment:
            self.export_result(results_dict)
        else:
            self.export_logs(run_logs, prob_id)

    def run_all(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        exp_name = f"logs_{self.dataset_str}_{self.args.batch_size}"
        exp_name += f"_{self.args.optimizer}_{self.args.learning_rate}"

        for prob_id in range(len(self.datasets)):
            self.run(prob_id=prob_id)

    def export_result(self, result_dict, prob_id):
        """Export SAT solving results to a file."""
        filename = os.path.join(self.save_dir / self.problem_name, "results.csv")
        df = pd.DataFrame.from_dict(results_dict)
        df = df.transpose() 
        df.to_csv(filename, sep="\t", index=False)

    def export_vector(self, stacked_vector, prob_id, tag="emb"):
        filename = os.path.join(self.save_dir / self.problem_name, tag + ".npy")
        with open(filename, "wb") as f:
            np.save(f, stacked_vector)

    def export_logs(self, log_dict, prob_id):
        filename = os.path.join(self.save_dir / self.problem_name, "run_logs.csv")
        df = pd.DataFrame.from_dict(log_dict)
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all()
