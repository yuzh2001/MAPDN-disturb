from mapdn.environments.var_voltage_control.disturbances import DisturbanceBase
import copy
import numpy as np
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl
class LoadChange(DisturbanceBase):
    """
    对负荷做扰动的类。

    disturbance_args: dict = {"multiplier": 2}
    """

    def start(self):
        super().start()
        self.env: VoltageControl = self.env # 激活python类型推断

        # update the record in the pandapower
        self.env.powergrid.sgen["p_mw"] = self.env.powergrid.sgen["p_mw"] * self.disturbance_args["multiplier"]
        self.env.powergrid.load["p_mw"] = self.env.powergrid.load["p_mw"] * self.disturbance_args["multiplier"]
        self.env.powergrid.load["q_mvar"] = self.env.powergrid.load["q_mvar"] * self.disturbance_args["multiplier"]

    def end(self):
        super().end()
