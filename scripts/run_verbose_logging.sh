lr=$1
OPTIMIZER=$2

python src/run.py \
    -d "data/counting_or_bl/*" -b 10000 \
    --loss_fn "l2_loss" \
    --lr $lr --optimizer ${OPTIMIZER} \
    --num_steps 100 \
    -nb \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "counting_fmcad_or_bl" \

python src/run.py \
    -d "data/counting_hard/*" -b 1000 \
    --loss_fn "l2_loss" \
    --lr $lr --optimizer ${OPTIMIZER} \
    --num_steps 100 \
    -nb \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "counting_fmcad_hard" \

# python src/run.py \
#     -d "data/counting2/blasted*" -b 10000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer ${OPTIMIZER} \
#     --num_steps 100 \
#     -nb \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "counting_blasted" \

# python src/run.py \
#     -d "data/counting2/tire-*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer ${OPTIMIZER} \
#     --num_steps 100 \
#     -nb \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "counting_tire" \

# python src/run.py \
#     -d "data/flat30/*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer ${OPTIMIZER} \
#     --num_steps 100 \
#     -nb \
#     # --wandb_entity "ucb-hcrl" \
#     # --wandb_project "gdsampler" \
#     # --wandb_group "flat30" \



# python src/run.py \
#     -d "data/counting2/*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer ${OPTIMIZER} \
#     --num_steps 100 \
#     -nb \
#     --num_experiments 100 \
    # --wandb_entity "ucb-hcrl" \
    # --wandb_project "gdsampler" \
    # --wandb_group "counting_first_100" \
 

# python src/run.py \
#     -d "data/beijing/*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer ${OPTIMIZER} \
#     --num_steps 10000 \
#     -nb \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "beijing" \

# python src/run.py \
#     -d "data/sat-comp-2023/*" -b 1000 \
#     --loss_fn "l2_loss" \
#     --lr $lr --optimizer ${OPTIMIZER} \
#     --num_steps 10000 \
#     -nb \
#     --wandb_entity "ucb-hcrl" \
#     --wandb_project "gdsampler" \
#     --wandb_group "sat-comp-2023" \
