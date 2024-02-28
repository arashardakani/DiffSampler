OPTIMIZER=$1

python src/jax/run.py \
    -d "data/prod/*1s*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "1e1,1.2e2,1.5e1,2e1" \
    --num_steps "10000" \
    --batch_size "2000,5000" \
    --wandb_group "report" \
