import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras

import play_guess_base_5_2 as guess_base
import game_verify_online_5_2 as game_verify
import traj_replayer_5_2 as replayer
import DNN_network_utils_5_2 as DNN_net

class PlayGuess_DNN(guess_base.PlayGuess_base):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_g='', learning_rate=0.00001, reload=False, net0_list=0, learning_amount=2000):
        super().__init__(filename_g=filename_g, learning_rate=learning_rate, learning_amount=learning_amount)
        self.replayer = replayer.TrajectoryReplayer(1e+8, state_shape=(5,54,), next_state_shape=(4,54*2))
        
        if ( reload == True ):
            self.guess_net = self.load_model()
            self.primay_net = self.guess_net
            self.secondary_net = [] #self.qt_net
            self.guess_net.summary()
            #self.qt_net.summary()
        else:
            input_size, input_shape = 5*54, (5*54,) #self.reshape_DNN_network()
            output_size1=3*54
            output_size2=1*54
            activation=tf2.nn.relu
            loss1=tf2.losses.mse
            loss2=tf2.losses.mse
            output_activation=tf2.nn.sigmoid # y y (= [0,1]. # activation=None. Q, doesn't use activation(). since softmax follows. BUT,y (= [0,1], sigmiod()更合适 and discard没有argsort问题（Q可能<1, unknow=0>Q is possible）
            ###############
            # net=q(s,a)
            ###############
            self.guess_net = DNN_net.build_guess_network(input_size, input_shape, hidden_layers, output_size1, output_size2,
                                                         activation, loss1, loss2, output_activation, learning_rate)
            self.primay_net = self.guess_net
            self.secondary_net = []
            #self.guess_net.summary() #print inside build_network
            
        return
    
    def shapes(self, learning_state3s):
        batch = learning_state3s.shape[0]
        return learning_state3s.reshape(batch, -1)
        
        
        
    ####################################
    # !! duplicate to DNN_base !!! start ....
    ####################################
    '''
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
    '''
    ####################################
    # !! duplicate to DNN_base !!!
    ####################################
    
    '''
    def integrity(self, state3s_batch):
        known_cards = np.sum(state3s_batch, axis=1)
        unknown_cards = np.logical_not(known_cards)[:,np.newaxis,:]
        
        #verify
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_17, known_cards)
        return unknown_cards
    
    def guess(self, state3s_batch, guess_env_callback):
        #state3s_batch shape=(batch-info(5,54))
        #output = (0,1) rather than fmt0/1/2/3. reshape in game_xx() who knows envs
        batch_size = state3s_batch.shape[0]
        
        state3s_batch0 = np.where(state3s_batch == 0, 0, 1)  #move fmt0/1/2/3 to (0,1)
        state3s_batch1 = state3s_batch0.reshape(batch_size, -1)
        #state3s_batch2 = state3s_batch0.reshape(batch_size, 5, 54)
        #c = (state3s_batch0 == state3s_batch2)
        guess_cards0 = self.guess_net.predict(state3s_batch1) #shape(batch-3-54)
        guess_cards1 = guess_cards0.reshape(batch_size, 3, 54)

        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_18, state3s_batch0, state3s_batch1, guess_cards0, guess_cards1)
        
        #每人牌的合理性, 总数，oindex分布 etc
        unknown_cards = self.integrity(state3s_batch0)

        #softmax
        #softmax_x = softmax(guess_cards1)
        
        x = guess_cards1 - np.max(guess_cards1, axis=1)[:,np.newaxis,:]  #+,-,*,/ : shape中只能有一个dim不等，才能操作
        exp_x = np.exp(x)
        exp_x_sum = np.sum(exp_x, axis=1)[:,np.newaxis,:]
        softmax_x = exp_x / exp_x_sum
        #shape: (batch,3,54) = (batch, 3, 54) * (batch, 1, 54)
        
        guess_cards2 = softmax_x * unknown_cards

        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_19, unknown_cards, guess_cards1, guess_cards2)

        #guess card from softmax
        index_012 = np.arange(3)
        guess_cards3 = np.zeros((batch_size, 3, 54))
        for i in range(batch_size): #loop=n*54. bad!!
            for j, possibles in enumerate(guess_cards2[i].T):
                if True == unknown_cards[i,0,j]:
                    guess_cards3[i, np.random.choice(index_012, p=possibles), j] = 1

        guess_cards = guess_env_callback(guess_cards3)  #translate from 0/1 to fmt2/3
        return guess_cards
    
    def guess_learning_single_round(self, state3s_batch, state3s_fraud_batch):
        length = state3s_batch.shape[0]
        dummy = np.zeros((length))
        history = []
        
        #here, state3s_batch and state3s_fraud_batch are different in shape.
        #1404bytes=1(5*54)+1(4*54*2). 200*1404 = 280kbytes
        stored_length = self.replayer.store(state3s_batch, dummy, dummy, dummy, state3s_fraud_batch)
        
        if self.learning_limit <= stored_length:
            learning_state3s, _, _, _, learning_state3s_fraud = self.replayer.sample_scope(0, -1)
            batch_size = learning_state3s.shape[0]
            history = self.guess_net.fit(learning_state3s.reshape(batch_size, -1), learning_state3s_fraud[:, 1:, 0:54].reshape(batch_size, -1), verbose=1, batch_size=256)
            self.replayer.remove_scope(0, -1)
        
        return history

    def guess_learning_multi_games(self):
        return
    
    def guess_learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #input state3: shape: round(=11)-player*batch-info(5, 54), or (4, 54*2)
        #action, reward shape: round(=11)-player*batch-info(0). no change in this method
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)
        return


    def save_models(self):
        super().save_model(self.guess_net, self.filename_g)

    '''