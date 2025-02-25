import os
import importlib

__API_REGISTRY__ = {}

def register_api(name:str):
    def register_model_fn(fn):
        if name in __API_REGISTRY__:
            return __API_REGISTRY__[name]
        __API_REGISTRY__[name] = fn
        return fn
    return register_model_fn

