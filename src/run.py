import glob
import logging
import os
import pathlib
import random
from datetime import datetime

import jax
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
from pysat.solvers import Solver
import wandb

import flags
import model.circuit_jax as circuit
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
        random.seed(self.args.seed)
        self.key = jax.random.PRNGKey(self.args.seed)
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.datasets = []
        self.dataset_str = ""
        if self.args.use_cpu:
            jax.config.update("jax_platform_name", "cpu")
        self.key = jax.random.PRNGKey(self.args.seed)
        self.do_wandb = self.args.wandb_entity is not None and not self.args.latency_experiment
        if self.do_wandb:
            assert self.args.wandb_project is not None
            config = {
                "batch_size": self.args.batch_size,
                "lr": self.args.learning_rate,
                "steps": self.args.num_steps,
                "train_data_path": self.args.dataset_path,
                "loss_fn": self.args.loss_fn,
                "optimizer": self.args.optimizer,
            }
            wandb_name = self.args.wandb_group + f"_{self.args.batch_size}_{self.args.optimizer}_{self.args.learning_rate}"
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=wandb_name,
                id=wandb_name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                resume="allow",
                group=self.args.wandb_group,
                # job_type=args.wandb_job_type,
                # tags=args.wandb_tags.split(","),
                config=config,
            )
            wandb.config.update(self.args)
        self._setup_problems()

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
            self.dataset_str = self.args.dataset_path.split("/")[1]
            self.results = {}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError

    def run_model(self, problem: CNF, prob_id: int = 0):
        solutions_found = []
        params, optimizer, literal_tensor, labels = circuit.init_problem(
            cnf_problem=problem,
            batch_size=self.args.batch_size,
            key=self.key,
            learning_rate=self.args.learning_rate,
            optimizer_str=self.args.optimizer,
        )
        if self.args.latency_experiment:
            (
                params,
                steps_ran,
                loss,
                elapsed_time,
                solutions_found,
            ) = circuit.run_back_prop(
                num_steps=self.args.num_steps,
                params=params,
                optimizer=optimizer,
                literal_tensor=literal_tensor,
                # labels=labels,
                loss_fn_str=self.args.loss_fn,
            )
        else:
            (
                params,
                steps_ran,
                loss,
                elapsed_time,
                solutions_found,
                log_dict,
            ) = circuit.run_back_prop_verbose(
                num_steps=self.args.num_steps,
                params=params,
                optimizer=optimizer,
                literal_tensor=literal_tensor,
                # labels=labels,
                loss_fn_str=self.args.loss_fn,
            )
            elapsed_time = 0
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
        self.results[prob_id].update(
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
        # wandb logging
        if self.do_wandb:
            for step in range(len(log_dict["loss"])):
                wandb.log(
                    {
                        "step": step,
                        "loss": log_dict["loss"][step],
                        "grad_norm": log_dict["grad_norm"][step],
                    }
                )
            wandb.finish()
        return 

    def run_baseline(self, problem: CNF, prob_id: int = 0):
        """Run the baseline solver.

        Args:
            problem: CNF problem to be solved
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        solver_names = self.args.baseline_name.split(",")

        for solver_name in solver_names:
            baseline = BaselineSolverRunner(
                cnf_problem=problem, solver_name=solver_name
            )
            baseline_elapsed_time, result, solver_solutions = baseline.run()
            logging.critical(f"Baseline {solver_name} Elapsed Time: {baseline_elapsed_time:.6f} seconds.")
            self.results[prob_id].update(
                {
                    f"baseline_{solver_name}_runtime": baseline_elapsed_time,
                    f"baseline_{solver_name}_result": result,
                }
            )
            if self.args.dump_solution:
                self.results[prob_id].update(
                    {f"baseline_{solver_name}_solution": solver_solutions}
                )
        return 

    def load_problem(self, prob_id: int):
        """Load a problem from the dataset.

        Args:
            prob_id (int): Index of the problem to be loaded.
        """
        logging.info(f"Loading problem from {self.datasets[prob_id]}")
        problem = CNF(from_file=self.datasets[prob_id])
        self.results[prob_id] = {
            "prob_desc": self.datasets[prob_id].split("/")[-1],
            "num_vars": problem.nv,
            "num_clauses": len(problem.clauses),
        }
        logging.info(f"num_vars: {problem.nv}")
        logging.info(f"num_clauses: {len(problem.clauses)}")
        return problem

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        problem = self.load_problem(prob_id)
        if not self.args.baseline_only:
            self.run_model(problem, prob_id)
        if not self.args.no_baseline:
            self.run_baseline(problem, prob_id)

    def run_all(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        for prob_id in range(len(self.datasets)):
            self.run(prob_id=prob_id)
        if self.args.latency_experiment:
            self.export_results()

    def export_results(self):
        """Export results to a file."""
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.problem_type}_{self.dataset_str}_{self.args.batch_size}_{self.args.loss_fn}"
        filename += f"_{self.args.optimizer}_{self.args.learning_rate}.csv"
        filename = os.path.join(self.save_dir, filename)
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all()
