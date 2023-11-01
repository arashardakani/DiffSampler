# Run SAT solving on pigeon hole problems"
GPU_ID=$1
lr=$2
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_jax.py -l -d "data/pigeon_hole_hard/*.cnf" -b 1000 --lr $lr
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_jax.py -l -d "data/pigeon_hole_hard/*.cnf" -b 10000 --lr $lr
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_jax.py -l -d "data/flat30/*.cnf" -b 1000 --lr $lr
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_jax.py -l -d "data/flat30/*.cnf" -b 10000 --lr $lr
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_jax.py -l -d "data/cbs_k3_n100_m403_b10/*.cnf" -b 1000 --lr $lr
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_jax.py -l -d "data/cbs_k3_n100_m403_b10/*.cnf" -b 10000 --lr $lr


# CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py --num_epochs 500 -v -l --dataset_path "data/pigeon_hole_hard/*SAT.cnf"
