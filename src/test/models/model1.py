from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten
from tensorflow.keras import Model


def create():
    i = Input(shape=(500,500,3))
    c = Conv2D(10,2)(i)
    c = Flatten()(c)
    c = Dense(2)(c)
    return Model(inputs=[i],outputs=[c])
