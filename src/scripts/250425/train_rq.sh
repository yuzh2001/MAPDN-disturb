SAVE_GROUP=$(date +%m%d/%H%M)
SAVE_GROUP=0425/1617

RUN_GROUP=$(date +%m%d/%H%M)
# uv run src/train/train.py --multirun train_config.alg=maddpg,mappo,matd3 train_config.alias=original  save_group=$SAVE_GROUP run_group=$RUN_GROUP
# uv run src/eval/eval.py eval_config.alg=mappo eval_config.alias=dis_new_q_weight_0.1_10t save_group=$SAVE_GROUP run_group=$RUN_GROUP
uv run src/eval/eval.py --multirun eval_config.alg=matd3 eval_config.alias=original,dis_new_q_weight_0.1_10t save_group=$SAVE_GROUP run_group=$RUN_GROUP