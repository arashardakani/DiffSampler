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
from tqdm import tqdm
import wandb

import flags
from model.initlialize_sat_prob import init_problem, init_optimizer
from model.gdsolve import gdsolve
from model.gdsolve_verbose import gdsolve_verbose



class SamplingRunner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        if not self.args.latency_experiment:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Args: {self.problem_type}")
        else:
            logging.basicConfig(level=logging.CRITICAL)

        # set random seed
        random.seed(self.args.seed)
        self.key = jax.random.PRNGKey(self.args.seed)

        # setup the dataset
        self.datasets = []
        self.dataset_str = ""
        self.configs = []
        self._setup_experiment()
        # setup the save directory
        self.save_dir = pathlib.Path(__file__).parent.parent.parent / f"results_{self.args.wandb_group}"
        save_dir_str = "{dataset}_{optimizer}_{timestamp}".format(
            dataset=self.dataset_str,
            optimizer=self.args.optimizer,
            timestamp=datetime.now().strftime("%Y%m%d"),
        )
        self.save_dir = self.save_dir / save_dir_str / "latency" if self.args.latency_experiment else self.save_dir / save_dir_str / "logging"
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
                "train_data_path": self.args.dataset_path,
                "optimizer": self.args.optimizer,
            }

    def _setup_experiment(self):
        """Setup the problems and configurations for the experiment."""
        if self.problem_type == "sat":
            assert self.args.dataset_path is not None
            dataset_path = os.path.join(
                pathlib.Path(__file__).parent.parent.parent, self.args.dataset_path
            )
            self.datasets = sorted(glob.glob(dataset_path), key=os.path.getsize)
            if self.args.num_experiments > 0:
                self.datasets = self.datasets[: self.args.num_experiments]
            self.dataset_str = self.args.dataset_path.split("/")[1]
            self.results = {}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError
        
        # setup the configurations
        learning_rates = self.args.learning_rate.split(",")
        num_steps = self.args.num_steps.split(",")
        batch_sizes = self.args.batch_size.split(",")
        if self.args.optimizer == "adam":
            b1_list = self.args.b1.split(",")
            b2_list = self.args.b2.split(",")
            momenta = [f"{b1},{b2}" for b1 in b1_list for b2 in b2_list]
        elif self.args.optimizer == "sgd":
            momenta = self.args.momentum.split(",")
        else:
            raise NotImplementedError

        # list of all combinations of configurations
        self.configs = [
            {
                "lr": lr,
                "ns": ns,
                "bs": bs,
                "mom": m,
            }
            for lr in learning_rates
            for ns in num_steps
            for bs in batch_sizes
            for m in momenta
        ]

        

    def load_problem(self, prob_id: int):
        """Load a CNF problem from the dataset.

        Args:
            prob_id (int): Index of the problem to be loaded.
        """
        logging.info(f"Loading problem from {self.datasets[prob_id]}")
        problem = CNF(from_file=self.datasets[prob_id])
        # logging_dict = {
        #     "prob_desc": self.datasets[prob_id].split("/")[-1],
        #     "num_vars": problem.nv,
        #     "num_clauses": len(problem.clauses),
        # }
        logging.info(f"num_vars: {problem.nv}")
        logging.info(f"num_clauses: {len(problem.clauses)}")
        return problem


    def generate_expr_name(self, problem_name: str, config_id: int):
        """Generate a name for the experiment."""
        config_str = "_".join(
            [f"{k}={v}" for k, v in self.configs[config_id].items()]
        )
        return f"{problem_name}_config_{config_str}"

    def run(self, problem: CNF, problem_name: str, config_id: int = 0):
        """Run the experiment."""
        experiment_str = self.generate_expr_name(problem_name=problem_name, config_id=config_id)
        config_str = "_".join(
            [f"{k}={v}" for k, v in self.configs[config_id].items()]
        )
        logging.critical(f"Run: {problem_name}; config {experiment_str}")
        
        if self.do_wandb:
            wandb_init_config = {
                "project": self.args.wandb_project,
                "entity": self.args.wandb_entity,
                "name": f"{problem_name}_{config_str}",
                "id": f"{problem_name}_{config_str}" + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "resume": "allow",
                "group": self.args.wandb_group + f"_{problem_name}_{self.args.optimizer}_"
                + "_".join(self.args.wandb_tags.split(",")),
                "tags": self.args.wandb_tags.split(","),
                "config": self.wandb_config,
            }
        else:
            wandb_init_config = {}

        # run the model for a given problem and configuration
        run_logs, model_results = self.run_model(problem=problem, config_id=config_id, wandb_init_config=wandb_init_config)
        if not self.args.latency_experiment:
            self.export_logs(log_dict=run_logs, experiment_str=experiment_str)
        return model_results

    def run_all(self):
        """Run all the problems in the data set given as argument to the Runner."""
        for prob_id in range(len(self.datasets)):
            problem_name = self.datasets[prob_id].split("/")[-1].split(".cnf")[0]
            self.problem_name = problem_name
            pathlib.Path(self.save_dir / self.problem_name).mkdir(
                parents=True, exist_ok=True
            )
            problem = self.load_problem(prob_id)
            results_dict_list = []
            for config_id in range(len(self.configs)):
                result_dict = self.configs[config_id].copy()
                model_results = self.run(problem=problem, problem_name=problem_name, config_id=config_id)
                result_dict.update(model_results)
                results_dict_list.append(result_dict)
            self.export_result(results_dict_list=results_dict_list, experiment_str=problem_name)


    def export_result(self, results_dict_list, experiment_str: str):
        filename = os.path.join(self.save_dir / self.problem_name, f"results_{experiment_str}.csv")
        # import pdb; pdb.set_trace()
        df = pd.DataFrame.from_dict(results_dict_list)
        # df = df.transpose() 
        df.to_csv(filename, sep="\t", index=False)

    def export_vector(self, stacked_vector, prob_id, tag="emb"):
        filename = os.path.join(self.save_dir / self.problem_name, tag + ".npy")
        with open(filename, "wb") as f:
            np.save(f, stacked_vector)

    def export_logs(self, log_dict, experiment_str):
        filename = os.path.join(self.save_dir / self.problem_name, f"logs_{experiment_str}.csv")
        df = pd.DataFrame.from_dict(log_dict)
        df.to_csv(filename, sep="\t", index=False)

    def run_model(self, problem: CNF, config_id: int, wandb_init_config: dict = {}):
        solutions_found = []

        # unzip the configuration
        experiment_config_dict = self.configs[config_id]
        lr = float(experiment_config_dict["lr"])
        num_steps = int(experiment_config_dict["ns"])
        batch_size = int(experiment_config_dict["bs"])
        momentum = experiment_config_dict["mom"] # note: is string

        params, literal_tensor, key = init_problem(
            cnf_problem=problem,
            batch_size=batch_size,
            key=self.key,
        )
        optimizer = init_optimizer(
            optimizer_str=self.args.optimizer,
            learning_rate=lr,
            momentum=momentum,
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
                num_steps=num_steps,
                params=params,
                optimizer=optimizer,
                literal_tensor=literal_tensor,
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
                num_steps=num_steps,
                params=params,
                optimizer=optimizer,
                literal_tensor=literal_tensor,
                prng_key=key,
                original_clauses=problem.clauses,
                do_wandb=self.do_wandb,
                do_log_all=self.args.dump_all,
                wandb_init_config=wandb_init_config,
            )
            log_dict = verbose_log_dict
            elapsed_time = 0
            # if self.args.dump_all: # dump all embeddings and losses
            #     self.export_vector(all_params, prob_id, tag="emb")
            #     self.export_vector(all_losses, prob_id, tag="loss")
        logging.info("--------------------")
        logging.info("Differential model solving")
        logging.info(
            f"Elapsed Time: {elapsed_time:.6f} seconds" f" Ran for {steps_ran} steps"
        )
        num_solutions = len(solutions_found)
        if num_solutions:
            logging.info("Model solution verified")
        else:
            logging.info("No solution")
        result_dict.update(
            {
                "model_runtime": elapsed_time,
                "model_steps_ran": steps_ran,
                "num_solutions": num_solutions,
                "loss": loss,
                "soln_tput": num_solutions / elapsed_time,
                "execution_tput": steps_ran / elapsed_time,
            }
        )
        logging.critical(f"Model runtime: {elapsed_time:.6f} seconds, Num solutions: {num_solutions}")
        # TODO: add support for exporting raw solutions found
        # if self.args.dump_solution:
        #     self.results[prob_id].update(
        #         {"model_solution": self.model.get_input_weights(solutions_found[i])}
        #     )

        return log_dict, result_dict

if __name__ == "__main__":
    runner = SamplingRunner(problem_type="sat")
    runner.run_all()
