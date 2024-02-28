OPTIMIZER=$1

# assuming x8 GPUs
# TARGET: 1K solutions
# will run x3 momentum values for SGD
# python src/jax/run.py \
#     -d "data/90-10/*.cnf*" -l\
#     --optimizer $OPTIMIZER\
#     --lr "5e0,1e1,2e1,5e1" \
#     --num_steps "1000" \
#     --batch_size "1000" \
#     --wandb_group "report" \

python src/jax/run.py \
    -d "data/90-10/*-2-*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "2.0,2.2,2.5,2.75" \
    --b1 "0.9"\
    --b2 "0.99"\
    --num_steps "5000" \
    --batch_size "5000" \
    --wandb_group "report" \

python src/jax/run.py \
    -d "data/90-10/*-6-*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "2.0,2.2,2.5,2.75" \
    --b1 "0.9"\
    --b2 "0.99"\
    --num_steps "5000" \
    --batch_size "5000" \
    --wandb_group "report" \

python src/jax/run.py \
    -d "data/90-10/*-8-*.cnf*" -l\
    --optimizer $OPTIMIZER\
    --lr "2.0,2.2,2.5,2.75" \
    --b1 "0.9"\
    --b2 "0.99"\
    --num_steps "5000" \
    --batch_size "5000" \
    --wandb_group "report" \