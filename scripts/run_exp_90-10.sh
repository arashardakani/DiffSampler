OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD
python src/jax/run.py \
    -d "data/90-10/*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "5e0,1e1,2e1,5e1" \
    --num_steps "2000" \
    --batch_size "1000" \
    --wandb_group "report" \
