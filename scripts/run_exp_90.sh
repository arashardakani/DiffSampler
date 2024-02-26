OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD
python src/jax/run.py \
    -d "data/or/*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "1e-1,1e0,1e1,1e2" \
    --num_steps "1000" \
    --batch_size "200,2000" \
    --wandb_group "report" \
