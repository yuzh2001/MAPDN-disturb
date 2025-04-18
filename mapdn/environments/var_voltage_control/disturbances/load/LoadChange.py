from mapdn.environments.var_voltage_control.disturbances import DisturbanceBase


class LoadChange(DisturbanceBase):
    """
    对负荷做扰动的类。

    disturbance_args: dict = {"multiplier": 2}
    """

    def start(self):
        super().start()
        for ter in self.env.terrain:
            ter.fixtures[0].friction = self.disturbance_args["friction"]

    def end(self):
        DEFAULT_FRICTION = 2.5
        for ter in self.env.terrain:
            ter.fixtures[0].friction = DEFAULT_FRICTION
        super().end()
