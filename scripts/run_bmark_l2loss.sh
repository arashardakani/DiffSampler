# Run SAT solving on pigeon hole problems"
GPU_ID=$1
lr=$2
opt=$3
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py -l -d "data/pigeon_hole_hard/*.cnf" -b 1000 --loss_fn 'l2_loss' --lr $lr --optimizer $opt
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py -l -d "data/pigeon_hole_hard/*.cnf" -b 10000 --loss_fn 'l2_loss' --lr $lr --optimizer $opt
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py -l -d "data/flat30/*.cnf" -b 1000 --loss_fn 'l2_loss' --lr $lr --optimizer $opt
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py -l -d "data/flat30/*.cnf" -b 10000 --loss_fn 'l2_loss' --lr $lr --optimizer $opt
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py -l -d "data/cbs_k3_n100_m403_b10/*.cnf" -b 1000 --loss_fn 'l2_loss' --lr $lr --optimizer $opt
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py -l -d "data/cbs_k3_n100_m403_b10/*.cnf" -b 10000 --loss_fn 'l2_loss' --lr $lr --optimizer $opt
