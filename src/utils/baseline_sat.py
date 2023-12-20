from time import time

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.examples.genhard import PHP
import pycryptosat
from threading import Timer
import subprocess
import re

def interrupt(s):
    s.interrupt()
    
class BaselineSolverRunner(object):
    """Baseline SAT solver runner."""

    def __init__(self, cnf_problem: CNF, solver_name: str = "m22"):
        self.solver_name = solver_name
        self.cnf = cnf_problem
        if solver_name == 'cms' or solver_name == 'cryptominisat':
            self.solver = pycryptosat.Solver()
            self.solver.add_clauses(cnf_problem.clauses)
        elif solver_name == 'cmsgen':
            pass
        elif solver_name == 'unigen':
            pass
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
        if self.solver_name == 'cms' or self.solver_name == 'cryptominisat':
            start_time = time()
            solve_out = self.solver.solve()
            end_time = time()
            elapsed_time = end_time - start_time
            sat = solve_out[0]
            if sat:
                solutions = [int(v) for v in solve_out[1][1:]]
        elif self.solver_name == 'cmsgen':
            sat = False
            solutions = ""
            elapsed_time = 0
            try:
                self.cnf.to_file("input.cnf")
                p1 = subprocess.run("/tools/designs/kevinhe/cmsgen/build/cmsgen input.cnf --samplefile mysamples.out --samples 1000 --seed 0", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True, timeout=7200)
                # print(p1.stdout)
                match = re.search(r'Total time: (\d+\.\d+)', p1.stdout)
                if match:
                    elapsed_time = float(match.group(1))
                print(f"elapsed time {elapsed_time} s")
                solutions = ""
                with open('mysamples.out', 'r') as file:
                    solutions = file.read()
                sat = True
            except subprocess.TimeoutExpired:
                print("timeout") 

        elif self.solver_name == 'unigen':
            self.cnf.to_file("input.cnf")
            sat = False
            solutions = ""
            elapsed_time = 0
            try:
                p1 = subprocess.run("/tools/designs/kevinhe/unigen/build/unigen --input input.cnf --sampleout mysamples.out --samples 1000 --seed 0", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True, timeout=7200)
                # print(p1.stdout)
                match = re.search(r'Time to sample: (\d+\.\d+) s', p1.stdout)
                if match:
                    elapsed_time = float(match.group(1))
                print(f"elapsed time {elapsed_time} s")
                with open('mysamples.out', 'r') as file:
                    solutions = file.read()
                sat = True
            except subprocess.TimeoutExpired:
                print("timeout")
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
