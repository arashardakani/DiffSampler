# Run SAT solving on pigeon hole problems"
GPU_ID=1#
######Prod Problems
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10000 --learning_rate 12.5 --batch_size 7000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/prod-1s.cnf.gz.no_w.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 30000 --learning_rate 12.5 --batch_size 20000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/prod-1s.cnf.gz.no_w.cnf"

CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10000 --learning_rate 3 --seed  100 --batch_size 1 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/04B-1.cnf.gz.no_w.cnf"


######Tire Problems
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 2000 --learning_rate 100 --batch_size 1600 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-1.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 2500 --learning_rate 100 --batch_size 14000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-1.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 2750 --learning_rate 50 --batch_size 3000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-2.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 2750 --learning_rate 50 --batch_size 30000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-2.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25000 --learning_rate 45 --batch_size 2000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-3.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 30000 --learning_rate 25 --batch_size 2000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-4.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 30000 --learning_rate 25 --batch_size 20000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/tire-4.cnf"


######Blast Problems



#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 1250 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_1.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 12500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_1.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 125000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_1.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 1300 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_2.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 13000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_2.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 130000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_2.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 1300 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_3.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 13000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_3.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 550 --learning_rate 100 --batch_size 130000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_3.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 1500 --learning_rate 45 --batch_size 1250 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_1.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 1500 --learning_rate 45 --batch_size 12500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_1.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 1500 --learning_rate 45 --batch_size 125000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_1.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10000 --learning_rate 20 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_2.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10000 --learning_rate 20 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_2.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10000 --learning_rate 20 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_2.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 3500 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_even.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 3500 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_even.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 3500 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b14_even.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25000 --learning_rate 18 --batch_size 1200 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_even1.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25000 --learning_rate 18 --batch_size 12000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_even1.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25000 --learning_rate 18 --batch_size 1200 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_even2.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25000 --learning_rate 18 --batch_size 12000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_even2.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 40000 --learning_rate 18 --batch_size 1500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_even3.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 40000 --learning_rate 18 --batch_size 15000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_case_1_b12_even3.cnf"






#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 50000 --learning_rate 22.5 --batch_size 1 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/blasted_squaring6.cnf"


######### Or problems
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10 --learning_rate 100 --batch_size 1050 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10 --learning_rate 100 --batch_size 10500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10 --learning_rate 100 --batch_size 105000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-10.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 100 --batch_size 1050 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 100 --batch_size 10500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 100 --batch_size 105000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-20.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25 --learning_rate 30 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25 --learning_rate 30 --batch_size 11500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 40 --learning_rate 30 --batch_size 120000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-30.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 22000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 10 --learning_rate 50 --batch_size 500000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-50-10-7-UC-40.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-10.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-20.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1200 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 12000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 120000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-30.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1200 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 12000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 120000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-60-20-10-UC-40.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1300 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 13000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 130000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-10.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 1300 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 13000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 15 --learning_rate 50 --batch_size 130000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-20.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1200 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 12000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 135000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-30.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 25 --learning_rate 50 --batch_size 13000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 150 --learning_rate 50 --batch_size 1350000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-70-5-5-UC-40.cnf"



#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1050 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 10500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-10.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 105000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-10.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-20.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-20.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1050 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 10500 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-30.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 105000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-30.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-40.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-40.cnf"

#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-50.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-50.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-50.cnf"


#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 1100 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-60.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 11000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-60.cnf"
#CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 20 --learning_rate 50 --batch_size 110000 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/or-100-20-8-UC-60.cnf"


######### s15850a problems
##CUDA_VISIBLE_DEVICES=$GPU_ID python ../src/run.py --num_epochs 200000 --learning_rate 30 --batch_size 1 -v -l --dataset_path "/home/eecs/arash.ardakani/Gates/SamplerBenchmarks/s15850a_3_2.cnf"





