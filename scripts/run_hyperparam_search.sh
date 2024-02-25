python src/jax/run.py \
    -d "data/blasted/*.cnf*" \
    --optimizer "sgd" -l \
    --lr "2e1,3e1,5e2" \
    --num_steps "5000" \
    --batch_size "2000" \
    --momentum "0.0,0.9, 0.99" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "hp_search" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/prod/*.cnf*" -l \
    --optimizer ${OPTIMIZER} \
    --lr "1e1,1.2e1,2e1,3e1" \
    --num_steps "10000" \
    --batch_size "2000" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "hp_search" \
    --wandb_tags "seed=0"
