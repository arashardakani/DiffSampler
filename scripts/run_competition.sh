lr=$1

python src/run.py -l -d "data/sat-comp-2023/*" -b 1000 --loss_fn 'l2_loss' --lr $lr --optimizer 'adam' --num_steps 10000
