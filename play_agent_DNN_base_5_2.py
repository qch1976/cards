
import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras
import psutil
import copy
import play_agent_NN_base_5_2 as NN_base

class PlayAgentDNNBase(NN_base.PlayAgentNNBase):
    def __init__(self, learning_rate, epsilon=0.2, gamma=0.0, net0_list=0, name=['non-behavior']):
        super().__init__(learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name)
    
    '''    
    ####################################
    # reinforcement learning NON-related common methods
    ####################################
    def build_network(self, input_size, input_shape, hidden_layers, output_size,
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
        ydl = model.get_weights()
        return model

    def reshape_DNN_network(self):
        #better to match the CNN network. similar value scope of axis0 and axis1
        if 4 == self.fraud:
            net_size = 4*54*2    #
        else:
            net_size = 5*54      #
        return net_size, (net_size,)
        
    '''