OPTIMIZER=$1

python src/jax/run.py \
    -d "data/tire/*1*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "5e1,7.5e1,1e2,1.5e2" \
    --num_steps "2000,5000" \
    --batch_size "200,2000" \
    --wandb_group "report" \

python src/jax/run.py \
    -d "data/tire/*2*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "2e1,3e1,4e1,5e1,7.5e1" \
    --num_steps "3000,5000" \
    --batch_size "200,2000" \
    --wandb_group "report" \


python src/jax/run.py \
    -d "data/tire/*3*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "2e1,3e1,4e1,5e1,7.5e1" \
    --num_steps "10000,20000" \
    --batch_size "1000,5000" \
    --wandb_group "report" \


python src/jax/run.py \
    -d "data/tire/*4*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "1e1,2e1,3e1,4e1,5e1" \
    --num_steps "10000,20000" \
    --batch_size "1000,5000" \
    --wandb_group "report" \