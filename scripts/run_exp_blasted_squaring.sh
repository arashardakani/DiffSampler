OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD and x12 for Adam

# squaring*
python src/jax/run.py \
    -d "data/blasted/*squaring*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "0.5,0.8,1.0,1.5,2.0" \
    --num_steps "1000,5000" \
    --batch_size "200,500,2000,5000" \
    --wandb_group "report" \
