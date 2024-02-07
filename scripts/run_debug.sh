lr=100.0
OPTIMIZER="sgd"


python src/run.py \
    -d "data/counting_debug/*" -b 100 \
    --lr $lr --optimizer ${OPTIMIZER} \
    --num_steps 100 \
    -nb \
    # --wandb_entity "ucb-hcrl" \
    # --wandb_project "gdsampler" \
    # --wandb_group "debug" \
    # --wandb_tags "l2,logprob"