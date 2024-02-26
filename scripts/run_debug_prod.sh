OPTIMIZER=$1
DEVICES=$2
CUDA_VISIBLE_DEVICES=$DEVICES python src/jax/run.py \
    -d "data/prod/*.cnf*"\
    --optimizer $OPTIMIZER\
    --lr "1e1,1.2e1,2e1,3e1,1e2" \
    --num_steps "2000" \
    --batch_size "10" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"
