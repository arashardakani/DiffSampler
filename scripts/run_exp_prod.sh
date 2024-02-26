OPTIMIZER=$1

python src/jax/run.py \
    -d "data/prod/*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "1e0,5e0,8e0,1e1,1.2e2,1.5e1" \
    --num_steps "5000,10000" \
    --batch_size "2000,5000" \
    --wandb_group "report" \
