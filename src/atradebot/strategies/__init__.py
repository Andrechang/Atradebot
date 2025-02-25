import os
import importlib
from atradebot.strategies.strategy import Strategy

__MODEL_REGISTRY__ = {}

def get_strategy(name:str):
    return __MODEL_REGISTRY__[name]

def register_strategy(name:str):
    def register_model_fn(fn):
        if name in __MODEL_REGISTRY__:
            return __MODEL_REGISTRY__[name]
        if not issubclass(fn, Strategy):
            raise ValueError(f"Model ({name}: {fn.__name__}) must be a subclass of Strategy")
        __MODEL_REGISTRY__[name] = fn
        return fn
    return register_model_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module(f"atradebot.strategies.{model_name}")
        

