# Code based off https://github.com/jpsety/verilog2dimacs/blob/master/verilog2dimacs.py
import re
import sys
import argparse
from pysat.formula import CNF
from pysat.solvers import Solver
import csv

def main():
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-o", "--output", help = "Output cnf path")
    parser.add_argument("-i", "--input", help = "input verilog path")
    parser.add_argument("-c", "--constraints", help = "constraints csv input")
    args = parser.parse_args()

    with open(args.input,'r') as f:
        data = f.read()
        f.seek(0)
        lines = f.readlines()

    # print(data)
    net_map = {}
    variable_index = 1

    #---------- Grab inputs -------------
    inputs = ""
    pattern = r'input(.*?);'
    match = re.search(pattern, data, re.DOTALL)
    if match:
        inputs = match.group(1)
    inputs = inputs.replace("\n", "")
    inputs = inputs.replace(" ", "")
    input_list = inputs.split(",")
    input_set = set(input_list)
    
    net_map.update(dict([(item, index + variable_index) for index, item in enumerate(input_list)]))
    variable_index += len(input_list)

    #---------- Grab outputs -----------
    outputs = ""
    pattern = r'output(.*?);'
    match = re.search(pattern, data, re.DOTALL)
    if match:
        outputs = match.group(1)
    outputs = outputs.replace("\n", "")
    outputs = outputs.replace(" ", "")
    output_list = outputs.split(",")
    output_set = set(output_list)

    net_map.update(dict([(item, index + variable_index) for index, item in enumerate(output_list)]))
    variable_index += len(output_list)

    #---------- Grab wires -------------
    wires = ""
    pattern = r'wire(.*?);'
    match = re.search(pattern, data, re.DOTALL)
    if match:
        wires = match.group(1)
    wires = wires.replace("\n", "")
    wires = wires.replace(" ", "")
    wire_list = wires.split(",")
    wire_set = set(wire_list)

    net_map.update(dict([(item, index + variable_index) for index, item in enumerate(wire_list)]))
    variable_index += len(wire_list)

    # print(net_map)

    #---------- Generate clauses from assigns -----------
    clauses = []
    inverted_nets = set()
    assign_lines = [line.strip() for line in lines if re.compile(r'^\s*assign.*').match(line)]
    for line in assign_lines:
        pattern = r'assign\s+(.*?)\s*='
        match = re.search(pattern, line)
        assignee = match.group(1)
        pattern = r'=\s+(.*?)\s*(\||&)'
        match = re.search(pattern, line)
        operand1 = match.group(1)
        pattern = r'(\||&)\s*(.*?)\s*;'
        match = re.search(pattern, line)
        operand2 = match.group(2)
        operator = match.group(1) # | or &
        if operand1[0] == '~' and operand1 not in net_map:
            inverted_nets.add(operand1)
            net_map[operand1] = variable_index
            variable_index += 1
            clauses.append(f'{net_map[operand1]} {net_map[operand1[1:]]} 0') #Create not gate (~x0 ∨ ~x1) ∧ (x0 ∨ x1)
            clauses.append(f'-{net_map[operand1]} -{net_map[operand1[1:]]} 0')
        if operand2[0] == '~' and operand2 not in net_map:
            inverted_nets.add(operand2)
            net_map[operand2] = variable_index
            variable_index += 1
            clauses.append(f'{net_map[operand2]} {net_map[operand2[1:]]} 0') #Create not gate (~x0 ∨ ~x1) ∧ (x0 ∨ x1)
            clauses.append(f'-{net_map[operand2]} -{net_map[operand2[1:]]} 0')
        if operator == '|':
            exp = f'-{net_map[assignee]}'
            for inp in [operand1, operand2]:
                clauses.append(f'{net_map[assignee]} -{net_map[inp]} 0')
                exp += f' {net_map[inp]}'
            exp += ' 0'
            clauses.append(exp)
        elif operator == '&':
            exp = f'{net_map[assignee]}'
            for inp in [operand1, operand2]:
                clauses.append(f'-{net_map[assignee]} {net_map[inp]} 0')
                exp += f' -{net_map[inp]}'
            exp += ' 0'
            clauses.append(exp)
        else:
            raise ValueError(f"Script does not support {operator}, only & and |")

    #---------- Constrain circuit nets ----------
    with open(args.constraints,'r') as f:
        csvFile = csv.reader(f)
        for line in csvFile:
            print(line)
            if line[1] == '1':
                clauses.append(f"{net_map[line[0]]} 0")
            elif line[1] == '0':
                clauses.append(f"-{net_map[line[0]]} 0")
            else:
                raise ValueError(f"Constraint is {line[1]}. Must be 1 or 0")
    
    #---------- Write CNF in DIMACS Format -----------
    with open(args.output, "w") as file:
        file.write(f"p cnf {variable_index-1} {len(clauses)}\n")
        clauses = '\n'.join(clauses)
        file.write(clauses)

    cnf = CNF()
    cnf.from_file(args.output)
    s = Solver(name='g4')
    for clause in cnf.clauses:
        s.add_clause(clause)
    s.solve()
    solution = s.get_model()
    input_sol = zip(input_list, map(lambda x: 0 if x < 0 else 1, solution[:len(input_list)]))
    output_sol = zip(output_list, map(lambda x: 0 if x < 0 else 1, solution[len(input_list):len(input_list) + len(output_list)]))
    
    print(list(input_sol))


if __name__ == "__main__":
    main()