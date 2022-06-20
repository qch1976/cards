import time
import numpy as np
import tensorflow.compat.v2 as tf2
import psutil
from tensorflow import keras
import play_agent_NN_base_5_2 as NN_base

class PlayAgentCNN_base(NN_base.PlayAgentNNBase):
    def __init__(self, learning_rate, epsilon=0.2, gamma=0.0, net0_list=0, name=['non-behavior']):
        NN_base.PlayAgentNNBase.__init__(self, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name)
        if self.net0_list != 0:
            print("PlayAgentCNN_base: net0 id + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
    
    '''
    def build_CNN_input_and_network(self, input_shape, conv_filters_in, kernal_sizes_in, strides_in, regularizer=keras.regularizers.l2(1e-4)):
        #input_shape = (5, 54, 1)
        x = keras.Input(shape=input_shape)
        x0 = x
        for conv_filter, kernal_size, stride in zip(conv_filters_in, kernal_sizes_in, strides_in):
            z = keras.layers.Conv2D(conv_filter, kernal_size, strides=stride, padding='same',
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
            y = keras.layers.BatchNormalization()(z)
            x = keras.layers.ReLU()(y)
            #print(z, y, x)
        cnn_outputs = x
        return x0, cnn_outputs

    #replaced by residual_yes=False in residual()
    #def build_CNN_network(self, cnn_inputs, conv_filters_in, kernal_sizes_in, strides_in, regularizer=keras.regularizers.l2(1e-4)):
    #    x = cnn_inputs
    #    for conv_filter, kernal_size, stride in zip(conv_filters_in, kernal_sizes_in, strides_in):
    #        z = keras.layers.Conv2D(conv_filter, kernal_size, strides=stride, padding='same',
    #                                kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    #        y = keras.layers.BatchNormalization()(z)
    #        x = keras.layers.ReLU()(y)
    #        #print(z, y, x)
    #    cnn_outputs = x
    #    return cnn_outputs
    
    def build_CNN_residual_network(self, x, conv_filters_res, kernal_sizes_res, strides_res, regularizer=keras.regularizers.l2(1e-4), residual_yes=True):
        #activation='relu'
        for res_filter_module, res_kernal_sizes_module, res_strides_module in zip(conv_filters_res, kernal_sizes_res, strides_res):
            shortcut = x
            for i, (residual_filter, kernal_size, stride) in enumerate(zip(res_filter_module, res_kernal_sizes_module, res_strides_module)):
                z = keras.layers.Conv2D(residual_filter, kernal_size, strides=stride, padding='same', 
                                        kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
                y = keras.layers.BatchNormalization()(z)
                if i >= len(res_filter_module) - 1 and True == residual_yes: #disable residual, then it is pure CNN
                    y = keras.layers.Add()([shortcut, y])  #add(): must be same shape
                #x = keras.layers.Activation(activation)(y)  #same to keras.layers.ReLU()
                x = keras.layers.ReLU()(y)

        residual_cnn_output = x
        return residual_cnn_output

    def build_CNN_flatten_output(self, cnn_inputs, output_shape, conv_filter_out, kernal_size_out, stride_out, regularizer=keras.regularizers.l2(1e-4)):
        z = keras.layers.Conv2D(conv_filter_out, kernal_size_out, strides=stride_out, padding='same',
                                kernel_regularizer=regularizer, bias_regularizer=regularizer)(cnn_inputs)
        y = keras.layers.BatchNormalization()(z)
        x = keras.layers.ReLU()(y)
        flattens = keras.layers.Flatten()(x)
        qs = keras.layers.Dense(output_shape, activation=keras.activations.tanh)(flattens) # 
        return qs

    def build_CNN_softmax_output(self, cnn_inputs, output_shape, conv_filter_out, kernal_size_out, stride_out, regularizer=keras.regularizers.l2(1e-4)):
        logits = keras.layers.Conv2D(conv_filter_out, kernal_size_out, strides=stride_out, padding='same',
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(cnn_inputs)
        flattens = keras.layers.Flatten()(logits)
        denses = keras.layers.Dense(output_shape)(flattens) # can it be removed? has to:flattens shape=probs
        activation='relu'
        active = keras.layers.Activation(activation)(denses)  #same to keras.layers.ReLU()
        probs = keras.layers.Softmax()(active)
        return probs

    def reshape_CNN_network(self):
        #better to match the CNN network. similar value scope of axis0 and axis1
        if 4 == self.fraud:
            shape = (4, 54*2, 1)    #<== (18, 24, 1)
        else:
            shape = (5, 54, 1)   #<== (15, 18, 1)
        return shape

    #moved from MC_q.
    def build_CNN_model(self, x0, qs, learning_rate, regularizer=keras.regularizers.l2(1e-4)):
        loss = keras.losses.MSE
        optimizer = keras.optimizers.Adam(learning_rate)

        model = keras.Model(inputs=x0, outputs=qs)
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()
        return model

    def build_CNN_network(self, input_shape, conv_filters_in, kernal_sizes_in, strides_in, 
                      conv_filters_out, kernal_sizes_out, strides_out, residual_net_config, learning_rate=0.00001):
        x0, cnn_outputs1 = self.build_CNN_input_and_network(input_shape, conv_filters_in, kernal_sizes_in, strides_in)
        
        if {} == residual_net_config:
            cnn_outputs2 = cnn_outputs1
        else:
            #residual Q
            conv_filters_res = residual_net_config['conv_filters']
            kernal_sizes_res = residual_net_config['kernal_sizes']
            strides_res      = residual_net_config['strides']
            cnn_outputs2 = self.build_CNN_residual_network(cnn_outputs1, conv_filters_res, kernal_sizes_res, strides_res)            
            
        cnn_inputs = cnn_outputs2
        output_shape = 54
        qs = self.build_CNN_flatten_output(cnn_inputs, output_shape, conv_filters_out, kernal_sizes_out, strides_out)
        model = self.build_CNN_model(x0, qs, learning_rate)
        return model

    def build_Res_model(self, x0, pis, vs, learning_rate):
        model = keras.Model(inputs=x0, outputs=[pis, vs])

        #need not reshape
        #def categorical_crossentropy_2d(y_true, y_pred):
        #    labels = tf2.reshape(y_true, [-1, 54])  #(batch, 54)
        #    preds = tf2.reshape(y_pred, [-1, 54])
        #    print("YDL: loss shape: ", y_true.shape, y_pred.shape)
        #    return keras.losses.categorical_crossentropy(labels, preds)
        #
        
        loss = [keras.losses.categorical_crossentropy, keras.losses.MSE]
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        print("loss.shape ", loss[0], loss[1])
        model.summary()
        return model

    def build_Res_network(self, input_shape, input_net_config, residual_net_config, policy_net_config, v_net_config, learning_rate=0.0001):
        conv_filters_in = input_net_config['conv_filters']
        kernal_sizes_in = input_net_config['kernal_sizes']
        strides_in      = input_net_config['strides']

        conv_filters_res = residual_net_config['conv_filters']
        kernal_sizes_res = residual_net_config['kernal_sizes']
        strides_res      = residual_net_config['strides']

        conv_filter_pi_out = policy_net_config['conv_filter']
        kernal_size_pi_out = policy_net_config['kernal_size']
        stride_pi_out      = policy_net_config['stride']

        conv_filter_v_out = v_net_config['conv_filter']
        kernal_size_v_out = v_net_config['kernal_size']
        stride_v_out      = v_net_config['stride']

        x0, cnn_outputs1 = self.build_CNN_input_and_network(input_shape, conv_filters_in, kernal_sizes_in, strides_in)
        cnn_outputs2 = self.build_CNN_residual_network(cnn_outputs1, conv_filters_res, kernal_sizes_res, strides_res)
        output_dense = 54
        pis = self.build_CNN_softmax_output(cnn_outputs2, output_dense, conv_filter_pi_out, kernal_size_pi_out, stride_pi_out)
        output_dense = 1
        vs = self.build_CNN_flatten_output(cnn_outputs2, output_dense, conv_filter_v_out, kernal_size_v_out, stride_v_out)
        
        model = self.build_Res_model(x0, pis, vs, learning_rate)
        return model
    '''
    
#ydl = PlayAgentCNN_base()