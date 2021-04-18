from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Add
from tensorflow.keras import Model


def create():
    i = Input(shape=(500,500,3),name="input1")
    c1 = Conv2D(5,2,name="conv1",padding="same")(i)
    c2 = Conv2D(5,3,name="conv2",padding="same")(i)
    c3 = Conv2D(5,3,name="conv3",padding="same")(i)
    c = Add()([c1,c2,c3])
    return Model(inputs=[i],outputs=[c])