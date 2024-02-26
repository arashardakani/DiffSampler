
OPTIMIZER=$1
DEVICES=$2
CUDA_VISIBLE_DEVICES=$DEVICES python src/jax/run.py \
    -d "data/modexp-test/*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "1e-2,1e-1,1e0,1e1,1e2" \
    --num_steps "2000" \
    --batch_size "1000" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"