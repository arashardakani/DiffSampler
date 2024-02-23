OPTIMIZER="sgd"

python src/jax/run.py \
    -d "data/or/*.cnf*" \
    --optimizer ${OPTIMIZER} \
    --lr "1e-1,1e0,5e0,1e1,5e1,1e2" \
    --num_steps "100,500,1000" \
    --batch_size "1000,2000" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "init" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/blasted/*.cnf*" \
    --optimizer ${OPTIMIZER} \
    --lr "1e-1,1e0,5e0,1e1,5e1,1e2" \
    --num_steps "100,1000,5000" \
    --batch_size "1000,2000,5000,10000" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "init" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/tire/*.cnf*" \
    --optimizer ${OPTIMIZER} \
    --lr "1e-1,1e0,5e0,1e1,5e1,1e2" \
    --num_steps "100,1000,2000" \
    --batch_size "1000,2000,5000,10000" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "init" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/prod/*.cnf*" \
    --optimizer ${OPTIMIZER} \
    --lr "1e0,5e0,1e1,5e1,1e2" \
    --num_steps "1000,2000,5000" \
    --batch_size "250,1000,2500" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "init" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/modexp/*.cnf*" \
    --optimizer ${OPTIMIZER} \
    --lr "1e0,5e0,1e1,5e1,1e2" \
    --num_steps "1000,2000,5000" \
    --batch_size "250,1000,2500" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "init" \
    --wandb_tags "seed=0"

python src/jax/run.py \
    -d "data/hash/*.cnf*" \
    --optimizer ${OPTIMIZER} \
    --lr "1e0,5e0,1e1,5e1,1e2" \
    --num_steps "1000,2000,5000" \
    --batch_size "250,1000,2500" \
    --momentum "0.0,0.9" \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "init" \
    --wandb_tags "seed=0"