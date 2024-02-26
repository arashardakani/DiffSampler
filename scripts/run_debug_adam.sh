
python src/jax/run.py \
    -d "data/modexp-test/*.cnf*"\
    --optimizer "sgd"\
    --lr "1e-1,1e0" \
    --num_steps "10000" \
    --batch_size "10" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/modexp-test/*.cnf*"\
    --optimizer "adam"\
    --lr "1e-1,1e0," \
    --num_steps "10000" \
    --batch_size "10" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"