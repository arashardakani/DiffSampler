OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD and x12 for Adam

# b14*
python src/jax/run.py \
    -d "data/blasted/*b14_1.cnf*" -l\
    --lr "2e1,4e1,5e1,7.5e1" \
    --optimizer $OPTIMIZER\
    --num_steps "700" \
    --batch_size "150,1500" \
    --wandb_group "report" 
python src/jax/run.py \
    -d "data/blasted/*b14_2.cnf*" -l\
    --lr "2e1,4e1,5e1,7.5e1" \
    --optimizer $OPTIMIZER\
    --num_steps "1000" \
    --batch_size "150,1500" \
    --wandb_group "report" 
python src/jax/run.py \
    -d "data/blasted/*b14_3.cnf*" -l\
    --lr "4e1,5e1,6e1,7.5e1" \
    --optimizer $OPTIMIZER\
    --num_steps "1000" \
    --batch_size "150,1500" \
    --wandb_group "report" 
python src/jax/run.py \
    -d "data/blasted/*b14_even.cnf*" -l\
    --lr "5e0,1e1,1.5e1,2e1" \
    --optimizer $OPTIMIZER\
    --num_steps "2000" \
    --batch_size "150,1500" \
    --wandb_group "report" 
# b12*
python src/jax/run.py \
    -d "data/blasted/*b12*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "1e1,1.5e1,2e1,4e1,5e1" \
    --num_steps "1000,2000" \
    --batch_size "150,1500" \
    --wandb_group "report" \

# # squaring*
# python src/jax/run.py \
#     -d "data/blasted/*squaring*.cnf*" -l\
#     --optimizer $OPTIMIZER\
#     --lr "0.5,0.8,1.0,1.5,2.0" \
#     --num_steps "1000,5000" \
#     --batch_size "200,500,2000,5000" \
#     --wandb_group "report" \
