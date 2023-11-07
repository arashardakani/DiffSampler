import glob
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
from pysat.solvers import Solver

import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, KLDivLoss
from tqdm import tqdm

import flags
import model.circuit as circuit
from utils.latency import timer
from utils.baseline_sat import BaselineSolverRunner


class Runner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = 'cpu'
        if self.args.verbose:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Args: {self.problem_type} on {self.device}")
            logging.info(
                "\n".join([f"{k}: {v}" for k, v in self.args.__dict__.items()])
            )
        else:
            logging.basicConfig(level=logging.ERROR)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.model = None
        self.loss = None
        self.optimizer = None
        self.baseline = None
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.datasets = []
        self.dataset_str = ""
        self._setup_problems()

    def _setup_problems(self):
        """Setup the problem.
        Implementation will later be extended to support other problem types. (currnetly only SAT)
            - SAT: dataset will be either (1) .cnf files in args.dataset_path or (2) PySAT PHP problems.
        """
        if self.problem_type == "sat":
            if self.args.dataset_path is None:
                logging.info("No dataset found. Generating PySAT PHP problems.")
                self.datasets = range(4, 12)
                self.problems = [PHP(nof_holes=n) for n in self.datasets]
                self.datasets = [f"PHP_{n}" for n in self.datasets]
                self.dataset_str = "php_4_12"
            else:
                dataset_path = os.path.join(
                    pathlib.Path(__file__).parent.parent, self.args.dataset_path
                )
                self.datasets = sorted(glob.glob(dataset_path))
                self.problems = None
                self.dataset_str = self.args.dataset_path.split("/")[1]
            self.results = {}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError

    def _initialize_model(self, prob_id: int = 0):
        """Initialize problem-specifc model, loss, optimizer and input, target tensors
        for a given problem instance, e.g. a SAT problem (in CNF form).
        Note: must re-initialize the model for each problem instance.

        Args:
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """

        if self.problem_type == "sat":
            if not self.problems:
                logging.info(f"Loading problem from {self.datasets[prob_id]}")
                problem = CNF(from_file=self.datasets[prob_id])
            else:
                problem = self.problems[prob_id]
            logging.info("Building model")
            self.model = circuit.CNF2Circuit(
                cnf_problem=problem,
                use_pgates=self.args.use_pgates,
                batch_size=self.args.batch_size,
                device=self.device,
            )
            self.input = torch.LongTensor(range(self.args.batch_size)).to(self.device)
            self.target = torch.ones(
                self.args.batch_size,
                len(problem.clauses),
                requires_grad=False,
                device=self.device,
            )
            if self.args.loss_fn == "mse":
                self.loss = MSELoss(reduction="mean")
                self.loss_per_batch = MSELoss(reduction="none")
            elif self.args.loss_fn == "bce":
                self.loss = BCELoss(reduction="mean")
                self.loss_per_batch = BCELoss(reduction="none")
            elif self.args.loss_fn == "ce":
                self.loss = CrossEntropyLoss(reduction="mean")
                self.loss_per_batch = CrossEntropyLoss(reduction="none")
            elif self.args.loss_fn == "kl":
                self.loss = KLDivLoss(reduction="mean")
                self.loss_per_batch = KLDivLoss(reduction="none")
            else:
                raise NotImplementedError
            # self.optimizer = torch.optim.Adam(
            #     self.model.parameters(), lr=self.args.learning_rate
            # )
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=2e-6
            )
            self.results[prob_id] = {
                "prob_desc": self.datasets[prob_id].split("/")[-1],
                "num_vars": problem.nv,
                "num_clauses": len(problem.clauses),
            }
            logging.info(f"num_vars: {problem.nv}")
            logging.info(f"num_clauses: {len(problem.clauses)}")
        else:
            raise NotImplementedError

        self.model.to(self.device)
        self.input.to(self.device)
        self.target.to(self.device)
        self.epochs_ran = 0

    def run_back_prop_with_early_exit(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""
        target = self.target.to(self.device)
        for epoch in train_loop:
            outputs = self.model(self.input)
            loss = self.loss(outputs, target)
            loss_per_batch = self.loss_per_batch(outputs, target)
            # logging.info(f"Epoch {epoch}: loss = {loss:.6f}")
            if torch.any(torch.all(loss_per_batch < 1e-3, dim=1)):
                # logging.info(f"Early exit at epoch {epoch}")
                return loss_per_batch
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.train()
        return loss_per_batch

    def run_back_prop(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""
        target = self.target
        for epoch in train_loop:
            outputs = self.model(self.input)
            loss = self.loss(outputs, target).to(self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.train()
            logging.info(f"Epoch {epoch}: loss = {loss:.6f}")
            logging.info(f"Weights: {self.model.layers['emb'].get_weights()[0][0]}")
        return self.loss_per_batch(outputs, self.target)

    def run_baseline(self, problem: CNF, prob_id: int = 0):
        """Run the baseline solver.

        Args:
            problem: CNF problem to be solved
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        if "comp" in self.dataset_str:
            solver_name = "cd153"
        else:
            solver_name = "m22"
        self.baseline = BaselineSolverRunner(
            cnf_problems=problem, solver_name=solver_name
        )
        baseline_elapsed_time, result = self.baseline.run()
        logging.info("--------------------")
        logging.info(f"Baseline Solver: {solver_name}")
        logging.info(f"Baseline Elapsed Time: {baseline_elapsed_time:.6f} seconds.")
        logging.info(f"Result: {result}")
        self.results[prob_id].update(
            {
                "baseline_runtime": baseline_elapsed_time,
                "baseline_result": result,
            }
        )
        if self.args.dump_solution:
            self.results[prob_id].update(
                {"baseline_core": self.baseline.solver.get_core()}
            )

        solver_solution = self.baseline.solver.get_model()
        self.baseline.solver.delete()
        return solver_solution

    def _check_solution(self, problem: CNF, assignment: dict[int:int]):
        return all(
            [
                any([assignment[abs(i)] * i > 0 for i in clause])
                for clause in problem.clauses
            ]
        )

    def verify_solution(self, problem: CNF, solutions: list[list[int]]):
        """Verify the solutions found by model
        Args:
            problem: CNF problem to be solved
            solutions: list of solutions found by model
        """
        is_verified = False
        for i, solution in tqdm(enumerate(solutions), total=len(solutions)):
            assignment = {i + 1: solution[i] for i in range(len(solution))}
            is_verified = self._check_solution(problem, assignment)
            # verifier = Solver(name='lgl', bootstrap_with=problem)
            # for assignment in solution:
            #     verifier.add_clause([assignment])
            # is_verified = verifier.solve()
            if is_verified:
                logging.info("Model solution verified")
                break
            # verifier.delete()
        if not is_verified:
            logging.info("No solution found")
        return is_verified

    def run_model(self, problem: CNF, prob_id: int = 0):
        solutions_found = []
        if self.args.latency_experiment:
            train_loop = range(self.args.num_epochs)
            if self.args.early_exit:
                elapsed_time, losses = timer(self.run_back_prop_with_early_exit)(
                    train_loop
                )
            else:
                elapsed_time, losses = timer(self.run_back_prop)(train_loop)
            logging.info("--------------------")
            logging.info("NN model solving")
            logging.info(f"Elapsed Time: {elapsed_time:.6f} seconds")
        else:
            train_loop = (
                range(self.args.num_epochs)
                if self.args.verbose
                else tqdm(range(self.args.num_epochs))
            )
            losses = self.run_back_prop(train_loop)
        if self.args.loss_fn != "ce":
            losses = torch.mean(losses, dim=1)
        solutions = torch.topk(1 - losses, k=self.args.topk, dim=0).indices.tolist()
        solutions_found = [self.model.get_input_weights(s) for s in solutions]
        solutions_found = [
            [(-1) ** (1 - sol[i]) * (i + 1) for i in range(len(sol))]
            for sol in solutions_found
        ]
        self.results[prob_id].update(
            {
                "model_runtime": elapsed_time,
                # "model_epochs_ran": self.epochs_ran,
            }
        )
        return solutions_found

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        self._initialize_model(prob_id=prob_id)
        problem = CNF(from_file=self.datasets[prob_id])
        # run NN model solving
        solutions_found = self.run_model(problem, prob_id)

        # solver_solution = self.run_baseline(problem, prob_id)
        # is_verified = any([sol == solver_solution for sol in solutions_found])
        # if not is_verified:
        #     is_verified = self.verify_solution(problem, solutions_found)
        is_verified = self.verify_solution(problem, solutions_found)
        self.results[prob_id].update(
            {
                "model_result": is_verified,
            }
        )
        # run baseline solver
        solver_solution = self.run_baseline(problem, prob_id)
        if self.args.dump_solution:
            self.results[prob_id].update(
                {"model_solution": self.model.get_input_weights(solutions_found[i])}
            )

        logging.info("--------------------\n")

    def run_all_with_baseline(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        for prob_id in range(len(self.datasets)):
            self.run(prob_id=prob_id)
        if self.args.latency_experiment:
            self.export_results()

    def export_results(self):
        """Export results to a file."""
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.problem_type}_{self.dataset_str}_{self.args.num_epochs}"
        filename += "_early_exit" if self.args.early_exit else ""
        filename += f"_{self.args.loss_fn}_{self.args.learning_rate}_{self.args.batch_size}_{self.args.topk}.csv"
        filename = os.path.join(self.save_dir, filename)
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all_with_baseline()
