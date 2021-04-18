from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Add
from tensorflow.keras import Model


def create():
    i = Input(shape=(500,500,3),name="input1")
    c = Conv2D(10,2,name="conv1")(i)

    i1 = Input(shape=(499,499,10),name="input2")
    c1 = Conv2D(10,2,name="conv2")(i1)
    c1 = Conv2D(10,2,name="conv3")(c1)
    m1 = Model(inputs=[i1],outputs=[c1],name="model2")

    c = m1(c)
    c = Flatten(name="flatten")(c)
    c = Dense(2,name="dense")(c)
    return Model(inputs=[i],outputs=[c],name="model1")
