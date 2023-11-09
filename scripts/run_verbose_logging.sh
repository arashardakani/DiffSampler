lr=1.0


   
# python src/run.py \
#     -d "data/flat30/*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer "adam" \
#     --num_steps 10000 \
#     -nb \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "flat30" \

python src/run.py \
    -d "data/beijing/*" -b 1000 \
    --loss_fn "l2_loss" \
    --lr $lr --optimizer "adam" \
    --num_steps 10000 \
    -nb \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "beijing" \

# python src/run.py \
#     -d "data/sat-comp-2023/*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer "adam" \
#     --num_steps 10000 \
#     -nb \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "sat-comp-2023" \