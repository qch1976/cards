#based off version 3m_mt
#v5: due to env 5.0 parallel running optm. discard alg is only 'dump'
#v5.2: add 54 output with 54 headers 

import os
import time
import numpy as np
import pandas as pd
import random as rd
from enum import Enum
from random import shuffle as rshuffle
import copy
import psutil
#import tensorflow.compat.v2 as tf
#from tensorflow import keras

from tensorflow import keras
import tensorflow as tf   #v1.14

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
#config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def ydl_measure(y_true, y_pred):
    loss_b = -tf.reduce_sum(y_true * y_pred, axis=-1)
    return loss_b



class DiscardAgent_net6_base:  #6 steps
    def __init__(self, learning_rate, epsilon=0.2, gamma=0.0, flash_t=False, net0_list=0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.flash_t = flash_t
        self.net0_list = net0_list
        self.pid = psutil.Process().pid  # get self pid
        print("discard PID: ", self.pid)
        print("discard base init: net0 id ", id(self.net0_list), id(net0_list))
        if self.net0_list == net0_list:
            print("discard base init net0 same")
        else:
            print("discard base init net0 NOT same")
            
        if self.net0_list != 0:
            print("discard base init ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])

    def build_network(self, input_size, hidden_layers, output_size,
                      activation, loss, output_activation=None, learning_rate=0.01, 
                      metrics=['accuracy']): # 构建网络  [ydl_measure]

        model = keras.Sequential()
        model.add(keras.layers.Dense(units=input_size, input_shape=(input_size,), activation=activation))

        for layer, [hidden_size, dropout] in enumerate(hidden_layers):
            model.add(keras.layers.Dense(units=hidden_size, activation=activation))
            model.add(keras.layers.Dropout(dropout))

        model.add(keras.layers.Dense(units=output_size, activation=output_activation)) # 输出层
        #optimizer = tf.optimizers.Adam(lr=learning_rate) #tf v2.+. CPU and IP=ute 19
        optimizer = keras.optimizers.Adam(lr=learning_rate) #tf v1.14. GPU, IP=133
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics) #['accuracy']), his['acc']
        model.summary()
        ydl = model.get_weights()
        return model

    def decide_1(self, net, state2s):  #collect 6 cards once. state2 size MUST be 18
        oindex = np.where(state2s > 0)

        qs0 = net.predict(state2s)
        qs = qs0[oindex[0], oindex[1]].reshape(-1,18)  #oindex is action
        q_max_index = np.argsort(-qs)  #(-):bigger -> smaller
        qs0_max_oindex = np.argsort(-qs0)  #(-):bigger -> smaller

        oindex_x_6 = oindex[0].reshape(-1, 18)[:,0:6].reshape(-1)
        oindex_y_6 = q_max_index[:,0:6].reshape(-1)
        q_max_oindex = oindex[1].reshape(-1, 18)[oindex_x_6, oindex_y_6].reshape(-1, 6)

        qs0_x_8 = oindex[0].reshape(-1, 18)[:,0:8].reshape(-1)
        qs0_y_8 = qs0_max_oindex[:,0:8].reshape(-1)
        qs0_max_8 = qs0[qs0_x_8, qs0_y_8].reshape(-1, 8)

        return q_max_oindex, qs0_max_oindex[:,0:8], qs0_max_8

    def decide_6(self, net, state2s):
        batch_size = len(state2s)
        oindex = np.where(state2s > 0)  # '!=' => '>' due to -1, 10, 100
        #oindex_len = int(len(oindex[1])/batch_size) #axis=1
        
        #state = state2[np.newaxis]  #full_cards_onehot_like, batch=1
        actions0 = net.predict(state2s)  #oindex(54,), action0(1,54)->(54)
        
        actions1 = actions0[oindex[0], oindex[1]].reshape(batch_size, -1)  #oindex is action
        action_max_index = np.argmax(actions1, axis=1)  #(-):bigger -> smaller
        actions0_max_index = np.argsort(-actions0)  #(-):bigger -> smaller
        
        oindex_x_1 = np.arange(batch_size)
        oindex_y_1 = action_max_index
        actions = oindex[1].reshape(batch_size, -1)[oindex_x_1, oindex_y_1]

        actions0_x_8 = oindex[0].reshape(batch_size, -1)[:,0:8].reshape(-1)
        actions0_y_8 = actions0_max_index[:,0:8].reshape(-1)
        actions0_max_8 = actions0[actions0_x_8, actions0_y_8].reshape(-1, 8)

        return actions, actions0_max_index[:, 0:8], actions0_max_8
    

    def save_model(self, net, filename):
        net.save(filename)
        ydl = net.get_weights()

    def load_model(self, filename, my_loss=0):
        net = keras.models.load_model(filename, custom_objects={'my_loss': my_loss}) #{'ydl_loss': ydl_loss}, NAME MUST BE SAME
        ydl = net.get_weights()
        return net

    def ydl_random(): #0.00 - 0.99
        ydl0 = time.time()
        ydl1 = ydl0 *100
        ydl2 = ydl1-int(ydl1)
        return ydl2

    def sync_local_to_mp_net0(self, net, net0_list):
        if net0_list == 0:
            return

        net_weights = net.get_weights()
        #print("agent6 base:sync_local_to_mp_net0:id ", id(self.net0_list))
        ydl_t = time.time()  
        #print(ydl_t, "agent6 base:sync_local_to_mp_net0:before ", self.pid, 
        #                                                   net0_list[0][0][0][0], net0_list[0][0][0][1], 
        #                                                   net0_list[0][2][50][50], net0_list[0][4][126][2])
        #print(ydl_t, "agent6 base:sync_local_to_mp_net0:before2 ", self.pid,
        #                                                    net_weights[0][0][0], net_weights[0][0][1], 
        #                                                    net_weights[2][50][50], net_weights[4][126][2])

        #net0_list = net0_list_0[0] #deepcopy doesn't work if refer as net0_list

        net_local_np = np.array(net_weights)
        #verify
        net_local_shape = [net_local_np[i].shape for i in range(net_local_np.shape[0])]
        #print("sync_local_to_mp_net0 get_weights: ", net_local_shape)
        #print("sync_local_to_mp_net0 net0_list shape: ", net0_list[2])
        
        
        if net_local_shape == net0_list[2]:
            net0_list[1].acquire()
            net0_list[0] = copy.deepcopy(net_weights) #weight matrix
            net0_list[1].release()
            
            #verify
            net0_list_len = [np.array(net0_list[0][i]).shape for i in range(len(net0_list[0]))]
            #print("net0_list_len: ", net0_list_len)
            ydl_t = time.time()
            #print(time.time(), "agent6 base:sync_local_to_mp_net0:after ", self.pid,
            #                                                  net0_list[0][0][0][0], net0_list[0][0][0][1], 
            #                                                  net0_list[0][2][50][50], net0_list[0][4][126][2])
        else:
            print("agent6 base:sync_local_to_mp_net0:after , shape wrong, net0, local ", self.pid, net0_list[2], net_local_shape)
        
        return

    def sync_mp_net0_to_local(self, net0_list, net):
        if net0_list == 0:
            return
        
        net_weights = net.get_weights()
        #print("agent6 base:sync_mp_net0_to_local:id ", id(self.net0_list))
        ydl_t = time.time()
        #print(ydl_t, "agent6 base:sync_mp_net0_to_local:before ", self.pid, 
        #                                                   net_weights[0][0][0], net_weights[0][0][1], 
        #                                                   net_weights[2][50][50], net_weights[4][126][2])

        #print(ydl_t, "agent6 base:sync_mp_net0_to_local:before2 ", self.pid,
        #                                                    net0_list[0][0][0][0], net0_list[0][0][0][1], 
        #                                                    net0_list[0][2][50][50], net0_list[0][4][126][2])

        #verify
        net_local_np = np.array(net_weights)
        net_local_shape = [net_local_np[i].shape for i in range(len(net_local_np))]
        #print("sync_mp_net0_to_local get_weights: ", net_local_shape)

        net0_list_len = [np.array(net0_list[0][i]).shape for i in range(len(net0_list[0]))]
        #print("sync_mp_net0_to_local net0_list_len: ", net0_list_len)

        net0_list_np = [np.array(net0_list[0][i]) for i in range(len(net0_list[0]))]

        if net_local_shape == net0_list[2]:
            net0_list[1].acquire()
            net.set_weights(net0_list_np) #net0_list[0])
            net0_list[1].release()
        
            #verify
            net_weights = net.get_weights()
            ydl_t = time.time()
            #print(ydl_t, "agent6 base:sync_mp_net0_to_local:after ", self.pid,
            #                                                  net_weights[0][0][0], net_weights[0][0][1], 
            #                                                  net_weights[2][50][50], net_weights[4][126][2])
        else:
            print("agent6 base:sync_mp_net0_to_local:after , shape wrong, net0, local ", self.pid, net0_list[2], net_local_shape)


class DiscardAgent_net6_Qmax2(DiscardAgent_net6_base):  #6 steps
    def __init__(self, hidden_layers, filename_e, filename_t, learning_rate=0.001, epsilon=0.2, gamma=0.0, reload=False, flash_t=False, net0_list=0):
        super().__init__(learning_rate, epsilon=epsilon, gamma=gamma, flash_t=flash_t, net0_list=net0_list)

        self.filename_e = filename_e
        self.filename_t = filename_t
    
    
        #self.net0_list_agent = net0_list #never used
        print("DiscardAgent_net6_Qmax2 init: net0 id ", id(self.net0_list))
        if self.net0_list != 0:
            print("DiscardAgent_net6_Qmax2 init ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                                   self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])
        if ( reload == True ):
            print("DiscardAgent_net6_Qmax2 reload start", filename_e)
            self.qe_net = self.load_model(filename_e)
            self.qt_net = self.load_model(filename_t)
            print("DiscardAgent_net6_Qmax2 reload 2")
            self.qe_net.summary()
            self.qt_net.summary()

            print("DiscardAgent_net6_Qmax2 reload done")
            if self.net0_list != 0:
                print("DiscardAgent_net6_Qmax2 reloaded ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                                       self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])
                if (0 == self.net0_list[0][0][0][0] and 0 == self.net0_list[0][0][0][1]): #PR#32, To be tested
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("DiscardAgent_net6_Qmax2: first copy to net0")
                    self.sync_local_to_mp_net0(self.qe_net, self.net0_list)
        else:                
            input_size=54
            #hidden_layers = [[512, 0.2], [128, 0.2]]
            output_size=54
            activation=tf.nn.relu
            #loss=tf.losses.mse #tf v2.+. CPU and IP=ute 19
            loss=tf.losses.mean_squared_error #tf v1.14. GPU, IP=133
            output_activation=None
            #learning_rate = 0.01
            self.qe_net = self.build_network(input_size, hidden_layers, output_size,
            								 activation, loss, output_activation, learning_rate)
            self.qt_net = self.build_network(input_size, hidden_layers, output_size,
											 activation, loss, output_activation, learning_rate)
            
            if net0_list != 0 :
                print("DiscardAgent_net6_Qmax2: net0 id ", id(net0_list))
                print("DiscardAgent_net6_Qmax2:init: ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                                                        net0_list[0][2][50][50], net0_list[0][4][126][2])

                if (0 == net0_list[0][0][0][0] and 0 == net0_list[0][0][0][1]):
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("DiscardAgent_net6_Qmax2: first copy to net0")
                    self.sync_local_to_mp_net0(self.qe_net, net0_list)
                else:
                    print("DiscardAgent_net6_Qmax2: second+ copy to loca")
                    self.sync_mp_net0_to_local(net0_list, self.qe_net)
                
                #verify
                '''
                a, b, c, d = 7, 3, 0, 7
                ydl = copy.deepcopy(net0_list[0])
                
                ydl[0][0][0] = 7 #copy.deepcopy(a)
                ydl[0][0][1] = 3 #copy.deepcopy(b)
                ydl[2][50][50] = copy.deepcopy(c)
                ydl[4][126][2] = copy.deepcopy(d)
                net0_list[0] = copy.deepcopy(ydl)

                print("DiscardAgent_net6_Qmax2: net0 id2 ", id(net0_list))
                print("DiscardAgent_net6_Qmax2:init2: ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                                                        net0_list[0][2][50][50], net0_list[0][4][126][2])
                self.sync_mp_net0_to_local(net0_list, self.qe_net)
                '''
            self.qt_net.set_weights(self.qe_net.get_weights())
            
        return

    def decide_onego(self, state2s):
        q_max_oindex, action0_index, action0 = super().decide_1(self.qe_net, state2s)
        return q_max_oindex, action0_index, action0

    def pre_learn_dump(self, state2s, best_discards_oindexes, rewards):
        '''
        state2s = np.array(trajectory0[0][0]).reshape(1,54)
        best_discards_oindex = np.array(trajectory0[0][1])
        rewards = np.array(trajectory0[0][2])
        '''
        batch_size = state2s.shape[0]
        if True == self.flash_t:
            targets = np.zeros([batch_size, 54])  #backgroud with 0
        else:
            targets0 = self.qe_net.predict(state2s)
            state2s_1 = np.where(state2s>0, 1, 0)  #>0 mean in-hand or trump
            targets = targets0 * state2s_1  #clear the position that oindex not existing
        
        '''
        for i in range(batch_size):
            targets[i,best_discards_oindexes[i]] = rewards[i]
        '''
        x = np.repeat(np.arange(batch_size), 6)
        y = best_discards_oindexes.reshape(-1)
        z = rewards.reshape(-1)
        ydl3 = targets[x, y]
        targets[x, y] = z
        
        self.sync_mp_net0_to_local(self.net0_list, self.qe_net)
        history = self.qe_net.fit(state2s, targets, verbose=0, batch_size=6*128)
        self.sync_local_to_mp_net0(self.qe_net, self.net0_list)
        return history

    def save_models(self):
        super().save_model(self.qe_net, self.filename_e)
        super().save_model(self.qt_net, self.filename_t)

    def echo(self, words):
        print("discard: echo: ", words)
        return words, self.filename_t


class DiscardAgent_net6_Qmax2_54headers(DiscardAgent_net6_Qmax2):  #6 steps
    def __init__(self, hidden_layers, filename_e, filename_t, learning_rate=0.001, epsilon=0.2, gamma=0.0, reload=False, flash_t=False, net0_list=0):
        super(DiscardAgent_net6_Qmax2, self).__init__(learning_rate, epsilon=epsilon, gamma=gamma, flash_t=flash_t, net0_list=net0_list)

        self.filename_e = filename_e
        self.filename_t = filename_t
    
    
        #self.net0_list_agent = net0_list #never used
        print("DiscardAgent_net6_Qmax2_54headers init: net0 id ", id(self.net0_list))
        if self.net0_list != 0:
            print("DiscardAgent_net6_Qmax2_54headers init ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                                   self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])
        if ( reload == True ):
            print("DiscardAgent_net6_Qmax2_54headers reload start", filename_e)
            self.qe_net = self.load_model(filename_e)
            self.qt_net = self.load_model(filename_t)
            print("DiscardAgent_net6_Qmax2_54headers reload 2")
            self.qe_net.summary()
            self.qt_net.summary()

            print("DiscardAgent_net6_Qmax2_54headers reload done")
            if self.net0_list != 0:
                print("DiscardAgent_net6_Qmax2_54headers reloaded ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                                       self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])
                if (0 == self.net0_list[0][0][0][0] and 0 == self.net0_list[0][0][0][1]): #PR#32, To be tested
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("DiscardAgent_net6_Qmax2: first copy to net0")
                    self.sync_local_to_mp_net0(self.qe_net, self.net0_list)
        else:                
            input_size=54
            #hidden_layers = [[512, 0.2], [128, 0.2]]
            output_size=54
            activation=tf.nn.relu
            #loss=tf.losses.mse #tf v2.+. CPU and IP=ute 19
            loss=tf.losses.mean_squared_error #tf v1.14. GPU, IP=133
            output_activation=None
            #learning_rate = 0.01
            self.qe_net = self.build_network(input_size, hidden_layers, output_size,      #replaced by 54 headers net
                                               activation, loss, output_activation, learning_rate)
            self.qt_net = self.build_network(input_size, hidden_layers, output_size,
                                               activation, loss, output_activation, learning_rate)
            
            if net0_list != 0 :
                print("DiscardAgent_net6_Qmax2_54headers: net0 id ", id(net0_list))
                print("DiscardAgent_net6_Qmax2_54headers:init: ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                                                        net0_list[0][2][50][50], net0_list[0][4][126][2])

                if (0 == net0_list[0][0][0][0] and 0 == net0_list[0][0][0][1]):
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("DiscardAgent_net6_Qmax2_54headers: first copy to net0")
                    self.sync_local_to_mp_net0(self.qe_net, net0_list)
                else:
                    print("DiscardAgent_net6_Qmax2_54headers: second+ copy to loca")
                    self.sync_mp_net0_to_local(net0_list, self.qe_net)
                
                #verify
                '''
                a, b, c, d = 7, 3, 0, 7
                ydl = copy.deepcopy(net0_list[0])
                
                ydl[0][0][0] = 7 #copy.deepcopy(a)
                ydl[0][0][1] = 3 #copy.deepcopy(b)
                ydl[2][50][50] = copy.deepcopy(c)
                ydl[4][126][2] = copy.deepcopy(d)
                net0_list[0] = copy.deepcopy(ydl)

                print("DiscardAgent_net6_Qmax2: net0 id2 ", id(net0_list))
                print("DiscardAgent_net6_Qmax2:init2: ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                                                        net0_list[0][2][50][50], net0_list[0][4][126][2])
                self.sync_mp_net0_to_local(net0_list, self.qe_net)
                '''
            self.qt_net.set_weights(self.qe_net.get_weights())
            
        return

    #net with 54 sub-net, each has 1 output
    def build_network(self, input_size, hidden_layers, output_size,
                      activation, loss, output_activation=None, learning_rate=0.01, 
                      metrics=['accuracy'], regularizer=None): #keras.regularizers.l2(1e-4)): # 构建网络  [ydl_measure]


        ### input layer
        input_shape = (input_size,)
        x0 = keras.Input(shape=input_shape)
        x = keras.layers.Dense(units=input_size, input_shape=input_shape, activation=activation)(x0)
    
        for layer, [hidden_size, dropout] in enumerate(hidden_layers):
            y1 = keras.layers.Dense(units=hidden_size, kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
            #y2 = keras.layers.BatchNormalization()(y1)
            y2 = y1
            y3 = keras.layers.ReLU()(y2)
            x = keras.layers.Dropout(dropout)(y3)
    
        #output layers, *2
        comm_outputs = x
        qs = []
        for i in range(output_size):
            Q = keras.layers.Dense(units=1, activation=output_activation, kernel_regularizer=regularizer, bias_regularizer=regularizer)(comm_outputs) # 输出层
            qs.append(Q)
    
        ### model creation
        losses = [loss]*output_size
        optimizer = keras.optimizers.Adam(learning_rate)
        model = keras.Model(inputs=x0, outputs=qs)
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics) #['accuracy']), his['acc'])
        model.summary()
        ydl = model.get_weights()
        return model


    #net = 54*1. 54 sub net. each has 1 output
    def decide(self, net, state2s):  #collect 6 cards once. state2 size MUST be 18
        oindex = np.where(state2s > 0)

        qs0 = net.predict(state2s)   #list=[54,(n,1)]
        qs1 = np.array(qs0).reshape(54,-1)
        qs2 = qs1.T

        qs = qs2[oindex[0], oindex[1]].reshape(-1,18)  #oindex is action
        q_max_index = np.argsort(-qs)  #(-):bigger -> smaller
        qs0_max_oindex = np.argsort(-qs2)  #(-):bigger -> smaller

        oindex_x_6 = oindex[0].reshape(-1, 18)[:,0:6].reshape(-1)
        oindex_y_6 = q_max_index[:,0:6].reshape(-1)
        q_max_oindex = oindex[1].reshape(-1, 18)[oindex_x_6, oindex_y_6].reshape(-1, 6)

        qs0_x_8 = oindex[0].reshape(-1, 18)[:,0:8].reshape(-1)
        qs0_y_8 = qs0_max_oindex[:,0:8].reshape(-1)
        qs0_max_8 = qs2[qs0_x_8, qs0_y_8].reshape(-1, 8)

        return q_max_oindex, qs0_max_oindex[:,0:8], qs0_max_8


    def decide_onego(self, state2s):  #collect 6 cards once. state2 size MUST be 18
        q_max_oindex, action0_index, action0 = self.decide(self.qe_net, state2s)  #net is replaced by 54 headers net
        return q_max_oindex, action0_index, action0

    #net with 54 sub-net, each has 1 output
    def pre_learn_dump(self, state2s, best_discards_oindexes, rewards):
        batch_size = state2s.shape[0]
        if True == self.flash_t:
            targets = np.zeros([batch_size, 54])  #backgroud with 0
        else:
            targets0 = self.qe_net.predict(state2s)  #output=(54, n, 1)
            targets1 = np.array(targets0).reshape(54, -1)
            targets2 = targets1.T
            state2s_1 = np.where(state2s>0, 1, 0)  #>0 mean in-hand or trump
            targets = targets2 * state2s_1  #clear the position that oindex not existing
        
        x = np.repeat(np.arange(batch_size), 6)
        y = best_discards_oindexes.reshape(-1)
        z = rewards.reshape(-1)
        targets[x, y] = z
        
        targets_l = []
        for i in range(54):
            targets_l.append(targets[:,i][:,np.newaxis])
        
        self.sync_mp_net0_to_local(self.net0_list, self.qe_net)
        history = self.qe_net.fit(state2s, targets_l, verbose=0, batch_size=6*128)
        self.sync_local_to_mp_net0(self.qe_net, self.net0_list)
        return history




class DiscardAgent_net6_Qmax2_CNN(DiscardAgent_net6_Qmax2):  #6 steps
    def __init__(self, hidden_layers, filename_e, filename_t, learning_rate=0.001, epsilon=0.2, gamma=0.0, reload=False, flash_t=False, net0_list=0):
        #hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}}
        
        super(DiscardAgent_net6_Qmax2, self).__init__(learning_rate, epsilon=epsilon, gamma=gamma, flash_t=flash_t, net0_list=net0_list)

        self.filename_e = filename_e
        self.filename_t = filename_t
    
        #self.net0_list_agent = net0_list #never used
        print("DiscardAgent_net6_Qmax2_CNN init: net0 id ", id(self.net0_list))
        if self.net0_list != 0:
            print("DiscardAgent_net6_Qmax2_CNN init ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                                   self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])
        if ( reload == True ):
            print("DiscardAgent_net6_Qmax2_CNN reload start", filename_e)
            self.qe_net = self.load_model(filename_e)
            self.qt_net = self.load_model(filename_t)
            print("DiscardAgent_net6_Qmax2_CNN reload 2")
            self.qe_net.summary()
            self.qt_net.summary()

            print("DiscardAgent_net6_Qmax2_CNN reload done")
            if self.net0_list != 0:
                print("DiscardAgent_net6_Qmax2_CNN reloaded ", self.net0_list[0][0][0][0], self.net0_list[0][0][0][1], 
                                                       self.net0_list[0][2][50][50], self.net0_list[0][4][126][2])
                if (0 == self.net0_list[0][0][0][0] and 0 == self.net0_list[0][0][0][1]): #PR#32, To be tested
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("DiscardAgent_net6_Qmax2_CNN: first copy to net0")
                    self.sync_local_to_mp_net0(self.qe_net, self.net0_list)
        else:                
            input_size=54
            #hidden_layers = [[512, 0.2], [128, 0.2]]
            output_size=54
            activation=tf.nn.relu
            #loss=tf.losses.mse #tf v2.+. CPU and IP=ute 19
            loss=tf.losses.mean_squared_error #tf v1.14. GPU, IP=133
            output_activation=None
            #learning_rate = 0.01
            
            input_net_config = hidden_layers['input_net']
            conv_filters_in = input_net_config['conv_filters']
            kernal_sizes_in = input_net_config['kernal_sizes']
            strides_in      = input_net_config['strides']

            self.qe_net = self.build_network(input_size, output_size,
                                               conv_filters_in, kernal_sizes_in, strides_in,
                                               activation, loss, output_activation, learning_rate)
            self.qt_net = self.build_network(input_size, output_size,
                                               conv_filters_in, kernal_sizes_in, strides_in,
                                               activation, loss, output_activation, learning_rate)
            
            if net0_list != 0 :
                print("DiscardAgent_net6_Qmax2_CNN: net0 id ", id(net0_list))
                print("DiscardAgent_net6_Qmax2_CNN:init: ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                                                        net0_list[0][2][50][50], net0_list[0][4][126][2])

                if (0 == net0_list[0][0][0][0] and 0 == net0_list[0][0][0][1]):
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("DiscardAgent_net6_Qmax2_CNN: first copy to net0")
                    self.sync_local_to_mp_net0(self.qe_net, net0_list)
                else:
                    print("DiscardAgent_net6_Qmax2_CNN: second+ copy to loca")
                    self.sync_mp_net0_to_local(net0_list, self.qe_net)

                #verify
                '''
                a, b, c, d = 7, 3, 0, 7
                ydl = copy.deepcopy(net0_list[0])
                
                ydl[0][0][0] = 7 #copy.deepcopy(a)
                ydl[0][0][1] = 3 #copy.deepcopy(b)
                ydl[2][50][50] = copy.deepcopy(c)
                ydl[4][126][2] = copy.deepcopy(d)
                net0_list[0] = copy.deepcopy(ydl)

                print("DiscardAgent_net6_Qmax2_CNN: net0 id2 ", id(net0_list))
                print("DiscardAgent_net6_Qmax2_CNN:init2: ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                                                        net0_list[0][2][50][50], net0_list[0][4][126][2])
                self.sync_mp_net0_to_local(net0_list, self.qe_net)
                '''
            self.qt_net.set_weights(self.qe_net.get_weights())
            
        return

    def build_network(self, input_size, output_size, 
                        conv_filters_in, kernal_sizes_in, strides_in, 
                        activation, loss, output_activation=None, learning_rate=0.01, 
                        metrics=['accuracy'], regularizer=keras.regularizers.l2(1e-4)): # 构建网络  [ydl_measure]

        #input_size = 54
        #input_shape = 6*9 
        input_shape = (6, 9, 1)
        x = keras.Input(shape=input_shape)
        x0 = x
        for conv_filter, kernal_size, stride in zip(conv_filters_in, kernal_sizes_in, strides_in):
            z = keras.layers.Conv2D(conv_filter, kernal_size, strides=stride, padding='same',
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
            y = keras.layers.BatchNormalization()(z)
            #y = z
            x = keras.layers.ReLU()(y)
            #print(z, y, x)
        cnn_outputs = x

        #output_size = 54
        flattens = keras.layers.Flatten()(cnn_outputs)
        qs = keras.layers.Dense(units=output_size, activation=output_activation)(flattens) #activation=output_activation

        loss = keras.losses.MSE
        optimizer = keras.optimizers.Adam(learning_rate)
    
        model = keras.Model(inputs=x0, outputs=qs)
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()
        return model


    #net = (6,9,1)
    def decide(self, net, state2s):  #collect 6 cards once. state2 size MUST be 18
        #state2s shape = (n,54)
        batch_size = state2s.shape[0]
        oindex = np.where(state2s > 0)

        qs0 = net.predict(state2s.reshape(batch_size, 6, 9, 1))   #input=(n, 54, 1); output=(n, 54)
        qs = qs0[oindex[0], oindex[1]].reshape(-1,18)  #oindex is action
        q_max_index = np.argsort(-qs)  #(-):bigger -> smaller
        qs0_max_oindex = np.argsort(-qs0)  #(-):bigger -> smaller

        oindex_x_6 = oindex[0].reshape(-1, 18)[:,0:6].reshape(-1)
        oindex_y_6 = q_max_index[:,0:6].reshape(-1)
        q_max_oindex = oindex[1].reshape(-1, 18)[oindex_x_6, oindex_y_6].reshape(-1, 6)

        qs0_x_8 = oindex[0].reshape(-1, 18)[:,0:8].reshape(-1)
        qs0_y_8 = qs0_max_oindex[:,0:8].reshape(-1)
        qs0_max_8 = qs0[qs0_x_8, qs0_y_8].reshape(-1, 8)
        
        return q_max_oindex, qs0_max_oindex[:,0:8], qs0_max_8


    def decide_onego(self, state2s):  #collect 6 cards once. state2 size MUST be 18
        q_max_oindex, action0_index, action0 = self.decide(self.qe_net, state2s)  #net is replaced by 54 headers net
        return q_max_oindex, action0_index, action0


    def pre_learn_dump(self, state2s, best_discards_oindexes, rewards):
        #state2s shape = (n,54)
        batch_size = state2s.shape[0]
        if True == self.flash_t:
            targets = np.zeros([batch_size, 54])  #backgroud with 0
        else:
            targets0 = self.qe_net.predict(state2s.reshape(batch_size, 6, 9, 1))  #input=(n, 6, 9, 1); output=(n, 54)
            state2s_1 = np.where(state2s>0, 1, 0)  #>0 mean in-hand or trump
            targets = targets0 * state2s_1  #clear the position that oindex not existing
        
        x = np.repeat(np.arange(batch_size), 6)
        y = best_discards_oindexes.reshape(-1)
        z = rewards.reshape(-1)
        targets[x, y] = z
        
        self.sync_mp_net0_to_local(self.net0_list, self.qe_net)
        history = self.qe_net.fit(state2s.reshape(batch_size, 6, 9, 1), targets, verbose=0, batch_size=6*128)
        self.sync_local_to_mp_net0(self.qe_net, self.net0_list)
        return history
