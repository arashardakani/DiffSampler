from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.examples.genhard import PHP

from utils.latency import timer


class BaselineSolverRunner(object):
    """Baseline SAT solver runner."""
    def __init__(self, cnf_problems: list[CNF], solver_name: str = "m22"):
        self.cnf_problems = cnf_problems
        self.solver_name = solver_name
        self.solver = Solver(
            name=self.solver_name,
            bootstrap_with=cnf_problems,
        )

    @timer
    def run(self):
        """Run the solver on the given problem.
        Assumes that the CNF problem to solve has been setup.
        """
        return self.solver.solve()
