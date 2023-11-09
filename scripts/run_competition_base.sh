lr=$1

python src/run.py -l -d "data/sat-comp-2023/*" -bo -bn "cms,mcb,cd15" --use_cpu
