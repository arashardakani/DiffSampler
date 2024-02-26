OPTIMIZER=$1

python src/jax/run.py \
    -d "data/modexp-test/*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "1e-3,5e-3,1e-2,5e-2,1e-2,5e-1" \
    --num_steps "10000" \
    --batch_size "100" \
    --wandb_group "report" \
