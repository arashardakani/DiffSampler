# HWverification

You can run the code using:

CUDA_VISIBLE_DEVICES=0 python -W ignore Circuit_adder.py


The binary values are represented using -1 and 1. The index 0 in vectors denotes the LSB.



To run the pigeon-hole problems:

CUDA_VISIBLE_DEVICES=0 python pigeon_hole_problems_prob.py --cnf_name_or_path ../pigeon_hole_problems/pigeon_hole_4-SAT.cnf
