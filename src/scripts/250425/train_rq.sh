GROUP_NAME=$(date +%m%d/%H%M)
GROUP_NAME=0425/1617
# uv run src/train/train.py --multirun train_config.alg=maddpg,mappo,matd3 train_config.alias=original  group_name=$GROUP_NAME
uv run src/eval/eval.py eval_config.alg=mappo eval_config.alias=dis_new_q_weight_0.1_10t group_name=$GROUP_NAME
uv run src/eval/eval.py --multirun eval_config.alg=matd3 eval_config.alias=original,dis_new_q_weight_0.1_10t group_name=$GROUP_NAME