import torch
import yaml
import pickle
from datetime import datetime
from mapdn.models.model_registry import Model, Strategy
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
from mapdn.utilities.util import convert
from mapdn.utilities.tester import PGTester

import hydra
from omegaconf import OmegaConf
import rich

from eval_hydra_type import EvalHydraEntryConfig, EvalConfig

import wandb


@hydra.main(config_path="../configs/eval", config_name="case33.yaml", version_base=None)
def run(configs: EvalHydraEntryConfig):
    # 1. 检查配置是否合法
    OmegaConf.structured(EvalConfig(**configs.eval_config))
    # for disturbance in configs.disturbances:
    # OmegaConf.structured(DisturbanceConfig(**disturbance))

    # 2. 运行
    rich.print(OmegaConf.to_container(configs, resolve=True))
    argv = configs.eval_config
    global_prefix = "./mapdn"

    # load env args
    with open(f"{global_prefix}/args/env_args/{argv.env}.yaml", "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = argv.scenario
    env_config_dict["data_path"] = "/".join(data_path)

    # set the action range
    net_topology = argv.scenario
    if argv.scenario == "case33_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif argv.scenario == "case141_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif argv.scenario == "case322_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    # 3. 设置环境参数
    assert argv.mode in [
        "distributed",
        "decentralised",
    ], "Please input the correct mode, e.g. distributed or decentralised."
    env_config_dict["mode"] = argv.mode
    env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

    # for one-day test
    env_config_dict["episode_limit"] = 480

    # load default args
    with open(global_prefix + "/args/default.yaml", "r") as f:
        default_config_dict = yaml.safe_load(f)
    default_config_dict["max_steps"] = 480

    # load alg args
    with open(global_prefix + "/args/alg_args/" + argv.alg + ".yaml", "r") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]
        alg_config_dict["action_scale"] = env_config_dict["action_scale"]
        alg_config_dict["action_bias"] = env_config_dict["action_bias"]

    log_name = "-".join(
        [
            # argv.env,
            net_topology,
            # argv.mode,
            argv.alg,
            argv.voltage_barrier_type,
            argv.alias,
        ]
    )
    wandb_name = "-".join(
        [
            f"[{datetime.now().strftime('%m%d-%H%M')}]",
            net_topology,
            argv.mode,
            argv.alg,
            argv.voltage_barrier_type,
            argv.alias,
        ]
    )
    print(f"Now testing: {wandb_name}")
    alg_config_dict = {**default_config_dict, **alg_config_dict}

    # define envs
    if configs.disturbances:
        env = VoltageControl(env_config_dict, configs.disturbances)
    else:
        env = VoltageControl(env_config_dict)

    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    alg_config_dict["cuda"] = False
    args = convert(alg_config_dict)

    # 读取模型训练结果
    if argv.save_path[-1] == "/":
        save_path = argv.save_path
    else:
        save_path = argv.save_path + "/"

    LOAD_PATH = save_path + log_name + "/model.pt"
    print(f"Loading model from {LOAD_PATH}")

    model = Model[argv.alg]

    strategy = Strategy[argv.alg]

    if args.target:
        target_net = model(args)
        behaviour_net = model(args, target_net)
    else:
        behaviour_net = model(args)
    checkpoint = (
        torch.load(LOAD_PATH, map_location="cpu")
        if not args.cuda
        else torch.load(LOAD_PATH)
    )
    behaviour_net.load_state_dict(checkpoint["model_state_dict"])

    if strategy == "pg":
        test = PGTester(args, behaviour_net, env, argv.render)
    elif strategy == "q":
        raise NotImplementedError("This needs to be implemented.")
    else:
        raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

    runa = wandb.init(
        project="mapdn-eval",
        name=wandb_name,
        save_code=True,
        config=OmegaConf.to_container(configs, resolve=True),
    )
    wandb.define_metric("terminate_cnt", summary="min")

    if argv.test_mode == "single":
        # record = test.run(199, 23, 2) # (day, hour, 3min)
        # record = test.run(730, 23, 2) # (day, hour, 3min)
        record = test.run(argv.test_day, 23, 2)
        with open(
            "reproduction/eval/test_record_"
            + log_name
            + f"_day{argv.test_day}"
            + ".pickle",
            "wb",
        ) as f:
            pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
    elif argv.test_mode == "batch":
        record = test.batch_run(argv.eval_episodes)
        data = [v for v in record.values()]
        wandb.log(
            {
                "results_table": wandb.Table(
                    columns=["wandb_name"] + list(record.keys()),
                    data=[[wandb_name] + data],
                )
            }
        )
        runa.finish()
        with open(
            "reproduction/eval/test_record_"
            + log_name
            + "_"
            + argv.test_mode
            + ".pickle",
            "wb",
        ) as f:
            pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run()
