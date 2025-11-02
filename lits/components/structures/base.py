from typing import Generic, List, Tuple, TypeVar, Union
from dataclasses import dataclass

@dataclass
class Step:
    pass
@dataclass
class State:
    pass
@dataclass
class Action:
    pass

# 泛型类型变量：必须是 State 或其子类
StateT = TypeVar("StateT", bound=State)
ActionT = TypeVar("ActionT", bound=Action)
StepT = TypeVar("StepT", bound=Step)

@dataclass
class Trace(Generic[StepT]):
    steps: List[StepT]

    def add(self, step: StepT):
        self.steps.append(step)

# 类型别名：StateByStepList 表示由 StepT 构成的列表
StateByStepList = list[Union[StepT]]