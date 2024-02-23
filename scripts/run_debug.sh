
OPTIMIZER="sgd"


CUDA_VISIBLE_DEVICES=0 python src/jax/run.py \
    -d "data/debug_tire/*.cnf*" -l \
    --optimizer ${OPTIMIZER} \
    --lr "1e-1,1e0" \
    --num_steps "1" \
    --batch_size "1,2" \
    --momentum "0.0,0.9" \
    # --wandb_entity "ucb-hcrl" \
    # --wandb_project "gdsampler" \
    # --wandb_group "debug" \
    # --wandb_tags "seed=0,momentum=0.0"