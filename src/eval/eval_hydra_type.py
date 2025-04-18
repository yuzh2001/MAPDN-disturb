from dataclasses import dataclass
from typing import List, Any, Optional
from enum import Enum

class AlgorithmType(Enum):
    coma = "coma"
    iddpg = "iddpg"
    ippo = "ippo"
    maddpg = "maddpg"
    mappo = "mappo"
    matd3 = "matd3"
    sqddpg = "sqddpg"

class EnvType(Enum):
    var_voltage_control = "var_voltage_control"

class ModeType(Enum):
    distributed = "distributed"
    decentralised = "decentralised"

class ScenarioType(Enum):
    case33_3min_final = "case33_3min_final"
    case141_3min_final = "case141_3min_final"
    case322_3min_final = "case322_3min_final"

class VoltageBarrierType(Enum):
    l1 = "l1"
    l2 = "l2"
    bowl = "bowl"

class TestModeType(Enum):
    single = "single"
    batch = "batch"

class DisturbanceType(Enum):
    load_change = "load_change"

@dataclass
class EvalConfig:
    save_path: str
    alg: AlgorithmType
    env: EnvType
    alias: str
    mode: ModeType
    scenario: ScenarioType
    voltage_barrier_type: VoltageBarrierType
    test_mode: TestModeType
    test_day: int
    render: bool

@dataclass
class DisturbanceConfig:
    type: DisturbanceType

    # for random disturbance
    is_random: bool
    random_probability: Optional[int]

    # for pre-defined disturbance
    start_at: int
    end_at: int

    disturbance_args: Any


@dataclass
class EvalHydraEntryConfig:
    eval_config: EvalConfig
    disturbances: Optional[List[DisturbanceConfig]]