OPTIMIZER="sgd"

python src/jax/run.py \
    -d "data/or/*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "1e-1,1e0,5e0,1e1,5e1,1e2,5e2" \
    --num_steps "1000,2000,5000,10000" \
    --batch_size "1000,2000,5000,10000" \
    --momentum "0.0,0.9" \

python src/jax/run.py \
    -d "data/blasted/*.cnf*" \
    --optimizer ${OPTIMIZER} -l \
    --lr "1e-1,1e0,5e0,1e1,5e1,1e2,5e2" \
    --num_steps "1000,2000,5000,10000" \
    --batch_size "1000,2000,5000,10000" \
    --momentum "0.0,0.9" \