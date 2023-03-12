import pandas as pd
from typing import Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Step(ABC):

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    

class Operation(ABC):

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)


@dataclass
class Pipeline(Operation):
    ops: List[Operation]

    def fit(self, *args, **kwargs):
        out = None
        for idx, op in enumerate(self.ops):
            if idx == 0:
                out = op(*args, **kwargs)
            else:
                out = op(**out)
        return out
    

@dataclass
class AddNumber(Step):
    number: Union[float, int]

    def transform(self, XY: pd.DataFrame, feature_dict: dict = None, *args, **kwargs):
        return XY + self.number, feature_dict


@dataclass
class SaveToFeaEngTable(Step):
    path: str

    def transform(self, XY: pd.DataFrame, feature_dict: dict = None, 
                  XY_sg: pd.DataFrame = None, XY_ospl: pd.DataFrame = None, 
                  *args, **kwargs):
        return pd.concat([XY, XY_sg, XY_ospl], ignore_index = True), feature_dict


@dataclass
class ReadFeaEngTable(Operation):
    sql: str

    def fit(self, XY: pd.DataFrame, feature_dict: dict, *args, **kwargs):
        XY_ospl = pd.DataFrame()
        XY_sg = pd.DataFrame({'a': [4, 5, 6], 'b': [9, 10, 11]})
        return {
            'XY': XY, 
            'feature_dict': feature_dict, 
            'XY_sg': XY_sg, 
            'XY_ospl': XY_ospl
        }


@dataclass
class FeaEngOperation(Operation):
    steps: List[Step]

    def fit(self, XY: pd.DataFrame, feature_dict: dict, *args, **kwargs):
        for step in self.steps:
            XY, feature_dict, *_ = step(XY, feature_dict, *args, **kwargs)
        return {
            'XY': XY, 
            'feature_dict': feature_dict
        }


if __name__ == '__main__':
    XY = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    feature_dict = {
        'features': []
    }

    out = Pipeline(ops = [
        ReadFeaEngTable(sql = "sql"),
        FeaEngOperation(steps = [
            AddNumber(1),
            AddNumber(2),
            SaveToFeaEngTable(path = "path"),
            AddNumber(1),
        ]),
    ]).fit(XY, feature_dict)

    print(out)