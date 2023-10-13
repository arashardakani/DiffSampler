import glob
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm

import flags
import model.circuit as circuit
from utils.latency import timer
from utils.baseline_sat import BaselineSolverRunner


class Runner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        if self.args.verbose:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Args: {self.problem_type} on {self.device}")
            logging.info(
                "\n".join([f"{k}: {v}" for k, v in self.args.__dict__.items()])
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        self.model = None
        self.loss = None
        self.optimizer = None
        self.solution_found = False
        self.baseline = None
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.dataset_str = ""

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        self._setup_problems()

    def _setup_problems(self):
        """Setup the problem.
        Implementation will later be extended to support other problem types. (currnetly only SAT)
            - SAT: dataset will be either (1) .cnf files in args.dataset_path or (2) PySAT PHP problems.
        """
        if self.problem_type == "sat":
            if self.args.dataset_path is None:
                logging.info("No dataset found. Generating PySAT PHP problems.")
                datasets = range(4, 12)
                self.problems = [PHP(nof_holes=n) for n in datasets]
                datasets = [f"PHP_{n}" for n in datasets]
                self.dataset_str = "php_4_12"
            else:
                dataset_path = os.path.join(pathlib.Path(__file__).parent.parent, self.args.dataset_path)
                datasets = sorted(glob.glob(dataset_path))
                self.problems = [CNF(from_file=path) for path in datasets]
                self.dataset_str = self.args.dataset_path.split('/')[1]
            self.baseline = BaselineSolverRunner(cnf_problems=self.problems)
            self.baseline_prover = BaselineSolverRunner(
                cnf_problems=self.problems, solver_name="lingeling"
            )
            self.results = {
                i: {"prob_desc": datasets[i].split('/')[-1]} for i in range(len(self.problems))
            }
            logging.info(f"Dataset used: {self.dataset_str}")
            # print(self.dataset_str)
            # import pdb; pdb.set_trace()
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
            # self.model = circuit.CombinationalCircuit(
            #     cnf_problem=self.problems[prob_id],
            #     use_pgates=self.args.use_pgates,
            #     batch_size=self.args.batch_size,
            #     device=self.device,
            # )
            self.model = circuit.CombCircuitWithClauseSAT(
                cnf_problem=self.problems[prob_id],
                use_pgates=self.args.use_pgates,
                batch_size=self.args.batch_size,
                device=self.device,
            )
            self.input = torch.LongTensor(range(self.args.batch_size)).to(self.device)  # batch size 1
            # self.target = torch.ones(1, requires_grad=False, device=self.device)
            self.target = torch.ones(self.args.batch_size, len(self.problems[prob_id].clauses), requires_grad=False, device=self.device)
            self.solution_prob_func = lambda loss: (
                1.0 - torch.round(loss * 32.) / 32.
            ).item()
            if self.args.loss_fn == "mse":
                self.loss = MSELoss(reduction='sum')
            elif self.args.loss_fn == "ce":
                self.loss = CrossEntropyLoss()
            elif self.args.loss_fn == "bce":
                self.loss = BCELoss(reduction='sum')
            else:
                raise NotImplementedError
            # self.loss = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        else:
            raise NotImplementedError

        self.model.to(self.device)
        self.input.to(self.device)
        self.target.to(self.device)
        self.solution_found = False

        

    def _check_complete(self, epoch: int, loss: torch.Tensor):
        """Check if the solution has been found.
        Used for early stopping backpropagation (grad descent) if the solution has been found.
        """
        current_solution_prob = self.solution_prob_func(loss)
        # logging.info('Loss @ epoch %d: %f', epoch, loss)
        # logging.info('Probability of getting 1 as the output @ epoch %d: %f', epoch, current_solution_prob)
        if current_solution_prob == 1.0:
            self.solution_found = True
            return True

    def run_back_prop(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""
        for epoch in train_loop:
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.input)
            l = self.loss(outputs, self.target)
            if self._check_complete(loss=l, epoch=epoch):
                break
            l.backward()
            self.optimizer.step()
        return epoch

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        self._initialize_model(prob_id=prob_id)
        if self.args.latency_experiment:
            train_loop = range(self.args.num_epochs)
            elapsed_time, epochs_ran = timer(self.run_back_prop)(train_loop)
            logging.info("--------------------")
            logging.info("NN model solving")
            logging.info(
                f"Elapsed Time: {elapsed_time:.6f} seconds; stopped at epoch: {epochs_ran}."
            )
            logging.info("--------------------")
        else:
            train_loop = (
                range(self.args.num_epochs)
                if self.args.verbose
                else tqdm(range(self.args.num_epochs))
            )
            epochs_ran = self.run_back_prop(train_loop)
        if self.solution_found:
            logging.info("Solution:"+"-".join([str(w) for w in self.model.get_input_weights()]))
        else:
            logging.info("No solution found.")
        self.results[prob_id].update(
            {
                "model_runtime": elapsed_time,
                "model_result": self.solution_found,
                "model_solution": self.model.get_input_weights(),
            }
        )

    def run_baseline(self, prob_id: int = 0):
        """Run the baseline solver.

        Args:
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        self.baseline._setup_problem(prob_id=prob_id)
        elapsed_time, result = self.baseline.run()
        logging.info("--------------------")
        logging.info("Baseline Solver: MiniSAT")
        logging.info(f"Elapsed Time: {elapsed_time:.6f} seconds.")
        logging.info(f"Result: {result}")
        if not result:
            logging.info(f"unsat core: {self.baseline.solver.get_core()}")
        self.results[prob_id].update(
            {
                "baseline_runtime": elapsed_time,
                "baseline_result": result,
                "baseline_core": self.baseline.solver.get_core(),
            }
        )
        logging.info("--------------------")
        if self.args.verify_solution:
            self.baseline_prover._setup_problem(prob_id=prob_id)
            elapsed_time, result = self.baseline_prover.run()
            logging.info("--------------------")
            logging.info("Baseline Solver: Lingeling")
            logging.info(f"Result: {result}")
            if not result:
                logging.info(f"Proof: {self.baseline_prover.solver.get_proof() is not None}")
            logging.info("--------------------")
            self.results[prob_id].update(
                {
                    "baseline_prover_runtime": elapsed_time,
                    "baseline_prover_result": result,
                    "baseline_prover_proof": self.baseline_prover.solver.get_proof(),
                }
            )

    def run_all_with_baseline(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        assert self.args.latency_experiment
        for prob_id in range(len(self.problems)):
            self.results[prob_id].update(
                {
                    "num_vars": self.problems[prob_id].nv,
                    "num_clauses": len(self.problems[prob_id].clauses),
                }
            )
            self.run(prob_id=prob_id)
            self.run_baseline(prob_id=prob_id)
        self.export_results()

    def export_results(self):
        """Export results to a file."""
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.problem_type}_{self.dataset_str}"
        filename += f"_{self.args.loss_fn}_{self.args.learning_rate}_{self.args.batch_size}.txt"
        filename = os.path.join(self.save_dir, filename)
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all_with_baseline()
