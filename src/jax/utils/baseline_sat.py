from time import time

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.examples.genhard import PHP
import pycryptosat
from threading import Timer


def interrupt(s):
    s.interrupt()


class BaselineSolverRunner(object):
    """Baseline SAT solver runner."""

    def __init__(self, cnf_problem: CNF, solver_name: str = "m22"):
        self.solver_name = solver_name
        if solver_name == "cms" or solver_name == "cryptominisat":
            self.solver = pycryptosat.Solver()
            self.solver.add_clauses(cnf_problem.clauses)
        else:
            self.solver = Solver(
                name=self.solver_name,
                bootstrap_with=cnf_problem,
            )

    def run(self):
        """Run the solver on the given problem.
        Assumes that the CNF problem to solve has been setup.
        """
        solutions = []
        if self.solver_name == "cms" or self.solver_name == "cryptominisat":
            start_time = time()
            solve_out = self.solver.solve()
            end_time = time()
            elapsed_time = end_time - start_time
            sat = solve_out[0]
            if sat:
                solutions = [int(v) for v in solve_out[1][1:]]
        else:
            timer = Timer(10000, interrupt, [self.solver])
            start_time = time()
            solve_out = self.solver.solve_limited(expect_interrupt=True)
            end_time = time()
            elapsed_time = end_time - start_time
            sat = solve_out
            if sat:
                solutions = [int(v > 0) for v in self.solver.get_model()]
            self.solver.delete()
        return elapsed_time, sat, solutions
