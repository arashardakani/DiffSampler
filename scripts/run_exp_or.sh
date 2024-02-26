OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD
python src/jax/run.py \
    -d "data/or/*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "5e-1,8e-1,1e0,2e0" \
    --num_steps "20" \
    --batch_size "150,1500" \
    --wandb_group "report" \
