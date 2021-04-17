from graphviz import Digraph
from tensorflow.keras import Model

import inspect
import graphviz


def attribute(name,value):
    return "{%s|%s}" % (name,str(value))

def node_str(name,**kargs):
    attrs_list_str = [attribute(name,value) for name,value in kargs.items()]
    return "{%s|{%s}" % (name,"|".join(attrs_list_str))

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def get_modified_args(layer):
    constructor_args = type(layer).__init__.__code__.co_varnames

    print(constructor_args)

def plot_model(model: Model):
    g = Digraph()
    ## Ecriture des noeuds
    for layers in model.layers:
        pass