import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras

import play_guess_base_5_2 as guess_base
import game_verify_online_5_2 as game_verify
import traj_replayer_5_2 as replayer
import CNN_network_utils_5_2 as CNN_net

class PlayGuess_CNN(guess_base.PlayGuess_base):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                 filename_g='', learning_rate=0.00001, reload=False, learning_amount=2000, residual=False):
        super().__init__(filename_g=filename_g, learning_rate=learning_rate, learning_amount=learning_amount)
        
        self.replayer = replayer.TrajectoryReplayer(1e+8, state_shape=(5,54), next_state_shape=(4,54*2))

        if ( reload == True ):
            self.guess_net = self.load_model()
            self.primay_net = self.guess_net
            self.secondary_net = [] #self.qt_net
            self.guess_net.summary()
        else:
            input_shape = (5,54,1) #self.reshape_CNN_network()
            input_net_config = hidden_layers['input_net']
            output_net_config = hidden_layers['output_net']
            
            conv_filters_in = input_net_config['conv_filters']
            kernal_sizes_in = input_net_config['kernal_sizes']
            strides_in      = input_net_config['strides']
            
            conv_filter_out = output_net_config['conv_filter']
            kernal_size_out = output_net_config['kernal_size']
            stride_out      = output_net_config['stride']

            if True == residual:
                residual_net_config = hidden_layers['residual_net']
            else:
                residual_net_config = {}

            ###############
            # net=q(s,a). dual output=(inhand, discard), inhand includes discard
            ###############
            self.guess_net = CNN_net.build_guess_CNN_network(input_shape, 
                                                             conv_filters_in, kernal_sizes_in, strides_in, 
                                                             conv_filter_out, kernal_size_out, stride_out,
                                                             residual_net_config, learning_rate)
            self.primay_net = self.guess_net
            self.secondary_net = []
            #self.guess_net.summary()

        return


    def shapes(self, learning_state3s):
        return learning_state3s[:,:,:,np.newaxis]

 