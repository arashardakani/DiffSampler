# Code based off https://github.com/jpsety/verilog2dimacs/blob/master/verilog2dimacs.py
import re
import sys
import argparse
from pysat.formula import CNF
from pysat.solvers import Solver
import csv
import os
def parse_bench(input, output, constraints):
    input_file = input
    with open(input_file,'r') as f:
        data = f.read()
        f.seek(0)
        lines = f.readlines()

    # print(data)
    net_map = {}
    variable_index = 1
    assign_map = {}
    input_list = []
    output_list = []
    for line in lines:
        if line[0] == '#':
            continue
        elif "INPUT" in line:
            pattern = r'INPUT\((.*?)\)'
            match = re.search(pattern, line)
            input = match.group(1)
            net_map[input] = variable_index
            variable_index += 1
            input_list.append(input)
        elif "OUTPUT" in line:
            input = ""
            pattern = r'OUTPUT\((.*?)\)'
            match = re.search(pattern, line)
            out = match.group(1)
            net_map[out] = variable_index
            variable_index += 1
            output_list.append(out)
        elif '=' in line:
            pattern = r'(.*?)\s*='
            match = re.search(pattern, line)
            assignee = match.group(1)
            net_map[assignee] = variable_index
            variable_index += 1
            pattern = r'\((.*)\)'
            match = re.search(pattern, line, re.DOTALL)
            operands = match.group(1)
            operands = operands.replace(" ", "")
            operands_list = operands.split(',')
            if len(operands_list) == 1:
                net_map[operands_list[0]] = variable_index
                variable_index += 1
            else:
                net_map.update(dict([(item, index + variable_index) for index, item in enumerate(operands_list)]))
                variable_index += len(operands_list)
            pattern = r'=\s*(and|nand|or|nor|xor|xnor|buf|not)\('
            match = re.search(pattern, line)
            gate = match.group(1)
            assign_map[assignee] = (gate, operands_list)
        elif line == '\n':
            continue
        else:
            raise ValueError(f"Unknown: {line}")  
    # print(input_list)
    # print(output_list)
    clauses = []
    for assignee, (gate, operands_list) in assign_map.items():
        if gate == 'and':
            exp = f'{net_map[assignee]}'
            for inp in operands_list:
                clauses.append(f'-{net_map[assignee]} {net_map[inp]} 0')
                exp += f' -{net_map[inp]}'
            exp += ' 0'
            clauses.append(exp)
        elif gate == 'nand':
            exp = f'-{net_map[assignee]}'
            for inp in operands_list:
                clauses.append(f'{net_map[assignee]} {net_map[inp]} 0')
                exp += f' -{net_map[inp]}'
            exp += ' 0'
            clauses.append(exp)
        elif gate == 'or':
            exp = f'-{net_map[assignee]}'
            for inp in operands_list:
                clauses.append(f'{net_map[assignee]} -{net_map[inp]} 0')
                exp += f' {net_map[inp]}'
            exp += ' 0'
            clauses.append(exp)
        elif gate == 'nor':
            exp = f'{net_map[assignee]}'
            for inp in operands_list:
                clauses.append(f'-{net_map[assignee]} -{net_map[inp]} 0')
                exp += f' {net_map[inp]}'
            exp += ' 0'
            clauses.append(exp)
        elif gate == 'not':
            clauses.append(f'{net_map[assignee]} {net_map[operands_list[0]]} 0')
            clauses.append(f'-{net_map[assignee]} -{net_map[operands_list[0]]} 0')
        elif gate == 'buf':
            clauses.append(f'-{net_map[operands_list[0]]} {net_map[assignee]} 0')
            clauses.append(f'{net_map[operands_list[0]]} -{net_map[assignee]} 0')
        elif gate == 'xor':
            while len(operands_list)>2:
                #create new net
                new_net = operands_list[-2]+operands_list[-1]
                net_map[new_net] = len(net_map)+1

                # add constraints
                clauses.append(f'-{net_map[new_net]} -{net_map[operands_list[-1]]} -{net_map[operands_list[-2]]} 0')
                clauses.append(f'-{net_map[new_net]} {net_map[operands_list[-1]]} {net_map[operands_list[-2]]} 0')
                clauses.append(f'{net_map[new_net]} -{net_map[operands_list[-1]]} {net_map[operands_list[-2]]} 0')
                clauses.append(f'{net_map[new_net]} {net_map[operands_list[-1]]} -{net_map[operands_list[-2]]} 0')

                # remove last 2 nets
                operands_list = operands_list[:-2]

                # insert new net
                operands_list.insert(0,new_net)

            # add constraints
            clauses.append(f'-{net_map[assignee]} -{net_map[operands_list[-1]]} -{net_map[operands_list[-2]]} 0')
            clauses.append(f'-{net_map[assignee]} {net_map[operands_list[-1]]} {net_map[operands_list[-2]]} 0')
            clauses.append(f'{net_map[assignee]} -{net_map[operands_list[-1]]} {net_map[operands_list[-2]]} 0')
            clauses.append(f'{net_map[assignee]} {net_map[operands_list[-1]]} -{net_map[operands_list[-2]]} 0')
        else:
            raise ValueError(f"{gate} not supported")
    
    #---------- Constrain circuit nets ----------
    if constraints != "":
        with open(constraints,'r') as f:
            csvFile = csv.reader(f)
            for line in csvFile:
                if line[1] == '1':
                    clauses.append(f"{net_map[line[0]]} 0")
                elif line[1] == '0':
                    clauses.append(f"-{net_map[line[0]]} 0")
                else:
                    raise ValueError(f"Constraint is {line[1]}. Must be 1 or 0")
    
    #---------- Arash's Constraints ----------
    module_name = input_file.split("/")[-1].replace(".bench", "")
    if module_name in ["c17"]:
        clauses.append(f"{net_map[output_list[-1]]} 0")
    elif module_name in ["c432"]:
        clauses.append(f"{net_map[output_list[0]]} 0")
        clauses.append(f"-{net_map[output_list[-1]]} 0")
    elif module_name in ["c880", "c1908", "c3540"]:
        clauses.append(f"{net_map[output_list[0]]} 0")
        clauses.append(f"-{net_map[output_list[15]]} 0")
        clauses.append(f"{net_map[output_list[-1]]} 0")
    elif module_name in ["c499", "c1355", "c6288"]:
        clauses.append(f"{net_map[output_list[0]]} 0")
        clauses.append(f"-{net_map[output_list[15]]} 0")
        clauses.append(f"{net_map[output_list[-1]]} 0") 
    else:
        clauses.append(f"{net_map[output_list[0]]} 0")
        clauses.append(f"-{net_map[output_list[15]]} 0")
        clauses.append(f"{net_map[output_list[31]]} 0")
        clauses.append(f"-{net_map[output_list[63]]} 0")
        clauses.append(f"{net_map[output_list[-1]]} 0")

    # elif module_name in ["c432"]:
    #     output = torch.cat(( outputs_list[0], outputs_list[-1] ), dim = -1)
    # elif module_name in ["c880", "c1908", "c3540"]:
    #     output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[-1] ), dim = -1)
    # elif module_name in ["c499", "c1355", "c6288"]:
    #     output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[-1] ), dim = -1)
    # else:
    #     output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[31], outputs_list[63], outputs_list[-1] ), dim = -1)
    
    # if module_name in ['c17']:
    #     target = torch.ones((batch_size, 1), device=device) #torch.cat((d[0],d[2]), dim = -1)
    # elif module_name in ['c432']:
    #     target = torch.cat((torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device) ), dim = -1)
        
    # elif module_name in ['c880', 'c1908', 'c3540']:
    #     target = torch.cat((torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device) ), dim = -1)
    # elif module_name in ['c499', 'c1355', 'c6288']:
    #     target = torch.cat((torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device) ), dim = -1)
    # else:
    #     target = torch.cat((torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device)), dim = -1)
    
    #---------- Write CNF in DIMACS Format -----------
    with open(output, "w") as file:
        file.write(f"p cnf {variable_index-1} {len(clauses)}\n")
        clauses = '\n'.join(clauses)
        file.write(clauses)
    
    return input_list, output_list


def parse_verilog(input, output, constraints):
    with open(input,'r') as f:
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
        print(line)
        pattern = r'assign\s+(.*?)\s*='
        match = re.search(pattern, line)
        assignee = match.group(1)
        pattern = r'=\s+(.*?)\s*;'
        match = re.search(pattern, line)
        operands = match.group(1)
        print(operands)
        if '&' in operands or '|' in operands:
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
        else:
            if operands == "1'b0":
                clauses.append(f"-{net_map[assignee]} 0")
            elif operands == "1'b1":
                clauses.append(f"{net_map[assignee]} 0")
            elif operands in net_map:
                clauses.append(f'-{net_map[operands]} {net_map[assignee]} 0')
                clauses.append(f'{net_map[operands]} -{net_map[assignee]} 0')
            else:
                raise ValueError(f"{operands} not found in inputs, outputs, or wires")

    #---------- Constrain circuit nets ----------
    if constraints != "":
        with open(constraints,'r') as f:
            csvFile = csv.reader(f)
            for line in csvFile:
                if line[1] == '1':
                    clauses.append(f"{net_map[line[0]]} 0")
                elif line[1] == '0':
                    clauses.append(f"-{net_map[line[0]]} 0")
                else:
                    raise ValueError(f"Constraint is {line[1]}. Must be 1 or 0")
                
    #---------- Write CNF in DIMACS Format -----------
    with open(output, "w") as file:
        file.write(f"p cnf {variable_index-1} {len(clauses)}\n")
        clauses = '\n'.join(clauses)
        file.write(clauses)
    
    return input_list, output_list

def solve_cnf(input_list, output_list, output):
    #---------- Read in CNF and solve -----------
    cnf = CNF()
    cnf.from_file(output)
    s = Solver(name='g4')
    for clause in cnf.clauses:
        s.add_clause(clause)
    s.solve()
    solution = s.get_model()
    input_sol = zip(input_list, map(lambda x: 0 if x < 0 else 1, solution[:len(input_list)]))
    output_sol = zip(output_list, map(lambda x: 0 if x < 0 else 1, solution[len(input_list):len(input_list) + len(output_list)]))
    
    print(list(input_sol))

def main():
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-i", "--input", default="", help = "Single input verilog path")
    parser.add_argument("-o", "--output",  default="", help = "Single output cnf path")
    parser.add_argument("-c", "--constraints", default="", help = "Single constraints csv input")
    parser.add_argument("--input_dir", default="", help = "Dir of input designs")
    parser.add_argument("--output_dir", default="", help = "Dir of output cnfs")
    parser.add_argument("--constraints_dir", default="", help = "Dir constraints csv input")
    parser.add_argument("--bench_file", action='store_true', default=False, help = "Use bench format instead of verilog. False by default")
    args = parser.parse_args()

    if args.input:
        if args.bench_file:
            input_list, output_list = parse_bench(args.input, args.output, args.constraints)
        else:
            input_list, output_list = parse_verilog(args.input, args.output, args.constraints)
        solve_cnf(input_list, output_list, args.output)
    elif args.input_dir:
        for filename in os.listdir(args.input_dir):
            input_file = os.path.join(args.input_dir, filename)
            if os.path.isfile(input_file):
                base_name, extension = os.path.splitext(filename)
                if not os.path.exists(args.output_dir): #Create output directory if it doesn't exist
                    os.makedirs(args.output_dir)
                output_file = os.path.join(args.output_dir, f"{base_name}.cnf")
                constraints_file = ""
                if args.constraints_dir:
                    constraints_file = os.path.join(args.constraints_dir, f"{base_name}.csv")
                    if not os.path.exists(constraints_file):
                        constraints_file = "" 
                        #raise ValueError(f"Constraints file for {base_name} does not exist in {args.constraints_dir}. Please include {constraints_file}") 
                if args.bench_file:
                    input_list, output_list = parse_bench(input=input_file, output=output_file, constraints=constraints_file)
                else:
                    input_list, output_list = parse_verilog(input=input_file, output=output_file, constraints=constraints_file)
                solve_cnf(input_list, output_list, output_file)
    else:
        raise ValueError(f"input file or input dir must be specified")


    


if __name__ == "__main__":
    main()