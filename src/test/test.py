import torch
import argparse
import yaml
import pickle

from mapdn.models.model_registry import Model, Strategy
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
from mapdn.utilities.util import convert
from mapdn.utilities.tester import PGTester

import hydra
from omegaconf import DictConfig, OmegaConf
import rich

@hydra.main(config_path="../configs/test_config", config_name="case33.yaml", version_base=None)
def run(configs: DictConfig):
    argv = configs.test_config
    rich.print(configs)
    global_prefix = "./mapdn"
    # load env args
    with open(global_prefix+"/args/env_args/"+argv.env+".yaml", "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = argv.scenario
    env_config_dict["data_path"] = "/".join(data_path)
    net_topology = argv.scenario

    # set the action range
    assert net_topology in ['case33_3min_final', 'case141_3min_final', 'case322_3min_final'], f'{net_topology} is not a valid scenario.'
    if argv.scenario == 'case33_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif argv.scenario == 'case141_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif argv.scenario == 'case322_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    assert argv.mode in ['distributed', 'decentralised'], "Please input the correct mode, e.g. distributed or decentralised."
    env_config_dict["mode"] = argv.mode
    env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

    # for one-day test
    env_config_dict["episode_limit"] = 480

    # load default args
    with open(global_prefix+"/args/default.yaml", "r") as f:
        default_config_dict = yaml.safe_load(f)
    default_config_dict["max_steps"] = 480

    # load alg args
    with open(global_prefix+"/args/alg_args/"+argv.alg+".yaml", "r") as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]
        alg_config_dict["action_scale"] = env_config_dict["action_scale"]
        alg_config_dict["action_bias"] = env_config_dict["action_bias"]

    log_name = f"{argv.env}-{net_topology}-{argv.mode}-{argv.alg}-{argv.voltage_barrier_type}-{argv.alias}"
    alg_config_dict = {**default_config_dict, **alg_config_dict}

    # define envs
    env = VoltageControl(env_config_dict)

    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    alg_config_dict["cuda"] = False
    args = convert(alg_config_dict)

    # define the save path
    if argv.save_path[-1] == "/":
        save_path = argv.save_path
    else:
        save_path = argv.save_path+"/"

    LOAD_PATH = save_path+log_name+"/model.pt"

    model = Model[argv.alg]

    strategy = Strategy[argv.alg]

    if args.target:
        target_net = model(args)
        behaviour_net = model(args, target_net)
    else:
        behaviour_net = model(args)
    checkpoint = torch.load(LOAD_PATH, map_location='cpu') if not args.cuda else torch.load(LOAD_PATH)
    behaviour_net.load_state_dict(checkpoint['model_state_dict'])

    # rich.print(args)
    if strategy == "pg":
        test = PGTester(args, behaviour_net, env, argv.render)
    elif strategy == "q":
        raise NotImplementedError("This needs to be implemented.")
    else:
        raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

    if argv.test_mode == 'single':
        # record = test.run(199, 23, 2) # (day, hour, 3min)
        # record = test.run(730, 23, 2) # (day, hour, 3min)
        record = test.run(argv.test_day, 23, 2)
        with open('reproduction/eval/test_record_'+log_name+f'_day{argv.test_day}'+'.pickle', 'wb') as f:
            pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
    elif argv.test_mode == 'batch':
        record = test.batch_run(10)
        with open('reproduction/eval/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
            pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    run()