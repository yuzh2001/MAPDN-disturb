import torch as th
from mapdn.utilities.util import translate_action, prep_obs
import numpy as np
import time
import rich
from tqdm import trange


class PGTester(object):
    def __init__(self, args, behaviour_net, env, render=False):
        self.env = env
        self.behaviour_net = (
            behaviour_net.cuda().eval() if args.cuda else behaviour_net.eval()
        )
        self.args = args
        self.device = th.device(
            "cuda" if th.cuda.is_available() and self.args.cuda else "cpu"
        )
        rich.print("Using device:", self.device)
        self.n_ = self.args.agent_num
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim
        self.render = render

    def run(self, day, hour, quarter):
        # reset env
        state, global_state = self.env.manual_reset(day, hour, quarter)

        # init hidden states
        last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

        record = {
            "pv_active": [],
            "pv_reactive": [],
            "bus_active": [],
            "bus_reactive": [],
            "bus_voltage": [],
            "line_loss": [],
        }

        record["pv_active"].append(self.env._get_sgen_active())
        record["pv_reactive"].append(self.env._get_sgen_reactive())
        record["bus_active"].append(self.env._get_res_bus_active())
        record["bus_reactive"].append(self.env._get_res_bus_reactive())
        record["bus_voltage"].append(self.env._get_res_bus_v())
        record["line_loss"].append(self.env._get_res_line_loss())

        for t in range(self.args.max_steps):
            if self.render:
                self.env.render()
                time.sleep(0.01)
            state_ = (
                prep_obs(state)
                .contiguous()
                .view(1, self.n_, self.obs_dim)
                .to(self.device)
            )
            action, _, _, _, hid = self.behaviour_net.get_actions(
                state_,
                status="test",
                exploration=False,
                actions_avail=th.tensor(self.env.get_avail_actions()),
                target=False,
                last_hid=last_hid,
            )
            _, actual = translate_action(self.args, action, self.env)
            reward, done, info = self.env.step(actual, add_noise=False)
            done_ = done or t == self.args.max_steps - 1
            record["pv_active"].append(self.env._get_sgen_active())
            record["pv_reactive"].append(self.env._get_sgen_reactive())
            record["bus_active"].append(self.env._get_res_bus_active())
            record["bus_reactive"].append(self.env._get_res_bus_reactive())
            record["bus_voltage"].append(self.env._get_res_bus_v())
            record["line_loss"].append(self.env._get_res_line_loss())
            next_state = self.env.get_obs()
            # set the next state
            state = next_state
            # set the next last_hid
            last_hid = hid
            if done_:
                break
        return record

    def batch_run(self, num_epsiodes=100):
        test_results = {}
        for epi in trange(num_epsiodes):
            # reset env
            state, global_state = self.env.reset()

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

            for t in range(self.args.max_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = (
                    prep_obs(state)
                    .contiguous()
                    .view(1, self.n_, self.obs_dim)
                    .to(self.device)
                )
                # rich.print("[State]", state_)
                action, _, _, _, hid = self.behaviour_net.get_actions(
                    state_,
                    status="test",
                    exploration=False,
                    actions_avail=th.tensor(self.env.get_avail_actions()),
                    target=False,
                    last_hid=last_hid,
                )
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps - 1
                next_state = self.env.get_obs()
                # rich.print("[Next State]", next_state)
                # rich.print("[Load]", self.env.powergrid.load["q_mvar"])
                for k, v in info.items():
                    if k == "sum_rewards":
                        continue
                    if "mean_" + k not in test_results.keys():
                        test_results["mean_" + k] = [v]
                    else:
                        test_results["mean_" + k].append(v)
                # set the next state
                state = next_state
                # set the next last_hid
                last_hid = hid
                if done_:
                    test_results["terminate_cnt"] = test_results.get("terminate_cnt", 0)
                    if t < self.args.max_steps - 2:
                        test_results["terminate_cnt"] += 1
                    break
            print(f"This is the test episode: {epi}")

            test_results["mean_sum_rewards"] = test_results.get("mean_sum_rewards", [])
            test_results["mean_sum_rewards"].append(info["sum_rewards"])
            test_results["mean_terminate_at_step"] = test_results.get(
                "mean_terminate_at_step", []
            )
            if t < self.args.max_steps - 2:
                test_results["mean_terminate_at_step"].append(t)
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        self.print_info(test_results)
        return test_results

    def print_info(self, stat):
        string = ["Test Results:"]
        for k, v in stat.items():
            string.append(k + f": mean: {v[0]:2.4f}, \t2std: {v[1]:2.4f}")
        string = "\n".join(string)
        print(string)
