import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras

####################################
# reinforcement learning NON-related common methods
####################################
def build_network(input_size, input_shape, hidden_layers, output_size,
                  activation, loss, output_activation=None, learning_rate=0.01, 
                  metrics=['accuracy']): # 构建网络  [ydl_measure]

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=input_size, input_shape=input_shape, activation=activation)) #(input_size[0],input_size[1])

    for layer, [hidden_size, dropout] in enumerate(hidden_layers):
        model.add(keras.layers.Dense(units=hidden_size, activation=activation))
        model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(units=output_size, activation=output_activation)) # 输出层
    optimizer = tf2.optimizers.Adam(lr=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics) #['accuracy']), his['acc']
    model.summary()
    return model

def build_guess_network(input_size, input_shape, hidden_layers, output_size1, output_size2,
                  activation, loss1, loss2, output_activation=None, learning_rate=0.01, 
                  metrics=['accuracy'], regularizer=keras.regularizers.l2(1e-4)): # 构建网络  [ydl_measure]

    ### input layer
    x0 = keras.Input(shape=input_shape)
    x = keras.layers.Dense(units=input_size, input_shape=input_shape, activation=activation)(x0)

    for layer, [hidden_size, dropout] in enumerate(hidden_layers):
        y1 = keras.layers.Dense(units=hidden_size, kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        y2 = keras.layers.BatchNormalization()(y1)
        y3 = keras.layers.ReLU()(y2)
        x = keras.layers.Dropout(dropout)(y3)

    #output layers, *2
    comm_outputs = x
    qs_inhand  = keras.layers.Dense(units=output_size1, activation=output_activation, kernel_regularizer=regularizer, bias_regularizer=regularizer)(comm_outputs) # 输出层
    qs_discard = keras.layers.Dense(units=output_size2, activation=output_activation, kernel_regularizer=regularizer, bias_regularizer=regularizer)(comm_outputs) # 输出层

    ### model creation
    loss = [loss1, loss2]
    optimizer = keras.optimizers.Adam(learning_rate)
    model = keras.Model(inputs=x0, outputs=[qs_inhand, qs_discard])
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics) #['accuracy']), his['acc'])
    model.summary()
    return model
    

def reshape_DNN_network(fraud):
    #better to match the CNN network. similar value scope of axis0 and axis1
    if 4 == fraud:
        net_size = 4*54*2    #
    else:
        net_size = 5*54      #
    return net_size, (net_size,)