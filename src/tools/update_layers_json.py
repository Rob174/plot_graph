from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, Add, \
    SeparableConv2D

from src.tools.SaveWriter import SaveWriter


def update_layer(layer):
    args = list(
        filter(lambda x: x != "self" and x != "kwargs" and x != "args", layer.__init__.__code__.co_varnames))
    with SaveWriter("../analyser/layers.json") as dico:
        dico[layer.__name__] = {"attributes":{},"format":{}}
        dico[layer.__name__]["attributes"] = {arg:"same" for arg in args}


if __name__ == "__main__":
    for layer in [BatchNormalization,Dropout,Activation,SeparableConv2D,Add,GlobalAveragePooling2D]:
        print("Layer")
        update_layer(layer)
