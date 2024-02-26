
python src/jax/run.py \
    -d "data/tire/*1*.cnf*" \
    --optimizer "adam" -l \
    --lr "8e1,1e2,1.2e2" \
    --num_steps "1000,3000" \
    --batch_size "1000,5000" \
    --wandb_group "report" \

python src/jax/run.py \
    -d "data/tire/*1*.cnf*" \
    --optimizer "sgd" -l \
    --lr "4e1,5e1,7.5e1" \
    --num_steps "1000,3000" \
    --batch_size "1000,5000" \
    --wandb_group "report" \

python src/jax/run.py \
    -d "data/tire/*2*.cnf*" \
    --optimizer "adam" -l \
    --lr "1e1,1.5e1,2e1" \
    --num_steps "3000" \
    --batch_size "500,5000" \
    --wandb_group "report" 

python src/jax/run.py \
    -d "data/tire/*2*.cnf*" \
    --optimizer "sgd" -l \
    --lr "2e1,4e1,5e1" \
    --num_steps "3000" \
    --batch_size "500,5000" \
    --wandb_group "report" 