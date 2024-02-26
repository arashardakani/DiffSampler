OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD and x12 for Adam

# b14*
python src/jax/run.py \
    -d "data/blasted/*b14_*.cnf*" -l\
    --lr "4e1,5e1,7.5e1" \
    --optimizer $OPTIMIZER\
    --num_steps "1000" \
    --batch_size "150,1500" \
    --wandb_group "report" 

# b12*
python src/jax/run.py \
    -d "data/blasted/*b12_1.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "5e0,1e1,1.5e1" \
    --num_steps "3000,5000" \
    --batch_size "200,5000" \
    --wandb_group "report" \

python src/jax/run.py \
    -d "data/blasted/*b12_2.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "1e0,5e0,1e1" \
    --num_steps "5000,10000" \
    --batch_size "200,5000" \
    --wandb_group "report" \