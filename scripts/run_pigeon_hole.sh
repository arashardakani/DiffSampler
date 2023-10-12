# Run SAT solving on pigeon hole problems"
GPU_ID=$1
# CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py --num_epochs 500 -v -l --dataset_path "data/flat30/*.cnf"
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py --num_epochs 500 -v -l --dataset_path "data/pigeon_hole_easy/*SAT.cnf"
# CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py --num_epochs 500 -v -l --dataset_path "data/pigeon_hole_hard/*SAT.cnf"
