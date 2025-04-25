GROUP_NAME=$(date +%m%d/%H%M)
uv run src/train/train.py --multirun train_config.alg=maddpg,mappo,matd3 train_config.alias=original,dis_0.1_10t group_name=$GROUP_NAME
uv run src/eval/eval.py --multirun eval_config.alg=maddpg,mappo,matd3 eval_config.alias=original,dis_0.1_10t group_name=$GROUP_NAME