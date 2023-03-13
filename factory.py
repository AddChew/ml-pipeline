import pandas as pd
from easydict import EasyDict
from typing import Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ObjectStore(EasyDict):
    pass


class Output:

    def __init__(self, *args, **kwargs):
        self.args = args[0] if len(args) == 1 else args
        self.kwargs = kwargs

    def __repr__(self):
        sep = ', '
        args = sep.join(map(str, self.args))
        kwargs = sep.join(f'{k} = {v}' for k, v in self.kwargs.items())
        return f'{self.__class__.__name__}({sep.join((args, kwargs)).strip(sep)})'
    
    def __call__(self):
        return self.args, self.kwargs
    

class Step(ABC):

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    def __call__(self, input: Output):
        args, kwargs = input()
        out = self.transform(*args, **kwargs)
        return Output(out)
    

@dataclass
class Operation:
    steps: List[Step]

    def fit(self, *args, **kwargs):
        out = Output(*args, **kwargs)
        for step in self.steps[:-1]:
            out = step(out)

        args, kwargs = out()
        last_step = self.steps[-1]
        return last_step.transform(*args, **kwargs)
            
    def __call__(self, input: Output) -> Output:
        args, kwargs = input()
        out = self.fit(*args, **kwargs)
        return Output(out)


@dataclass
class Pipeline:
    ops: List[Operation]

    def fit(self, *args, **kwargs):
        out = Output(*args, **kwargs)
        for op in self.ops[:-1]:
            out = op(out)

        args, kwargs = out()
        last_op = self.ops[-1]
        return last_op.fit(*args, **kwargs)
    

@dataclass
class ReadFeaEngTable(Step):
    sql: str
    objectstore: ObjectStore

    def transform(self, XY: pd.DataFrame, feature_dict: dict, *args, **kwargs):
        XY_ospl = pd.DataFrame()
        XY_sg = pd.DataFrame({'a': [4, 5, 6], 'b': [9, 10, 11]})
        self.objectstore.update({
            'XY_ospl': XY_ospl,
            'XY_sg': XY_sg
        })
        return XY, feature_dict
    

@dataclass
class AddNumber(Step):
    number: Union[float, int]

    def transform(self, XY: pd.DataFrame, feature_dict: dict = None, *args, **kwargs):
        return XY + self.number, feature_dict


@dataclass
class SaveToFeaEngTable(Step):
    path: str
    objectstore: ObjectStore

    def transform(self, XY: pd.DataFrame, feature_dict: dict = None, *args, **kwargs):
        XY_sg = self.objectstore.pop('XY_sg')
        XY_ospl = self.objectstore.pop('XY_ospl')
        return pd.concat([XY, XY_sg, XY_ospl], ignore_index = True), feature_dict


if __name__ == '__main__':
    XY = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    feature_dict = {
        'features': []
    }
    objectstore = ObjectStore()
    out = Pipeline(ops = [
        Operation(steps = [
            ReadFeaEngTable(sql = "sql", objectstore = objectstore),
            AddNumber(1),
            AddNumber(2),
            SaveToFeaEngTable(path = "path", objectstore = objectstore),
            AddNumber(1),
        ]),
    ]).fit(XY, feature_dict)
    print(out)