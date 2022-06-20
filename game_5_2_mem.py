#v4.2: update for TD serise implementations. but blocked due to imperfect iformation game theory
#      update .h5 file name, remove gameid(in InputParameters.read() only) for crossing competition. 缺点：init时，容易误删已有的.h5
#v5.0: aglin to env v5.0. parallel envs running optm
#    : support GPU. 
#v5.1: state3s_fraud
#      runtime checkpoints added
#      CNN added
#      fraud shape -> (4, 54*2). added 'played' info. has error. but online checking MISSED!
#v5.2: state refactory for diff-structured(state3, state3_fraud, and coming states)
#      reward alg changed. fmt[-1, 0.5, 1, 2] is obsolete
#      add TD with playback
#      guess class added. train in orund
##################################################
# ToDOs
# 1. state3, state2 in Game{} is useless, remove them
# 2. in CNN Q alg, input would be in range [-1, 1] so that the normalization works. as well as the loss=MSE
#    in DNN Q alg, there is no batchnormalization(). but the predict() result is no big
# 3. DONE. TD with and without fraud mode, replay mode. how to setup the experiecen priority?
# 4. DONE. reshape the network for CNN with similar value scope of axis0 and axis1
# 5. when init the agents, if the 'agent id' are same, should not create new instance. refer to eisting one 
##################################################
import os
import time
import numpy as np
import random as rd
import copy
import psutil
import sys, getopt
from multiprocessing import Process, Manager, set_start_method
#import gc
#gc.disable()

import tensorflow as tf
#in root@CentOS: apply all CPUs(28 cores) at ???: 
#in ute@Debian: apply all CPUs(28 cores) at ???: does 'root' impact that? A: no, both root and ute take CPU >20%
#BTW, sudo or su, doesn't make the conda env(. block running python discard_testbench...) A: 'conda init bash' will add conda env to root rc
#CPU mode: 5.6h(GPU-133 28 CPUs with 60%+) vs. 36h(ute 1 CPU=100%)
#TB verify in 'game.py' ????

#below 3 lines would make GPU 2-3% increase. but don't the efficiency is imprvoed or not. need meas the full time cost
tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_v2_tensorshape()

####################################################
# GPU enable NOTES:
# MUST: tf.compat.v1.keras.backend.set_session() at very beginning of py file, before any program code
# if move into GPU_prepare(), tf will report "no GPU device found' when tf init
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.95  #default = 100%(actual=~96%)
config.gpu_options.allow_growth = True  #按需分配
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#TBD: tf.test.is_gpu_available(). in console, it found and created the GPU. not verify in .py

#from absl import flags
def GPU_prepare(gpu_id):
    #verify
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("YDL1: GPU: ", gpus)

    #######################
    # no meaningful code in GPU_prepare()
    #######################
    
    # it is late if perform below lines here. MUST run them before any program code !!
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
    #config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    return







import play_agent_DNN_MC_q_5_2 as agent_1
import play_agent_DNN_MC_pi_5_2 as agent_2
import play_agent_DNN_TD_q_5_2 as agent_3
import play_agent_CNN_MC_q_5_2 as agent_6
import play_agent_CNN_MC_pi_5_2 as agent_7
import play_agent_CNN_TD_q_5_2 as agent_11

import config_5_2 as cfg
import CNN_network_utils_5_2 as CNN_net
import play_guess_CNN_5_2 as guess



#unify inputs: 
#1.render
#set at start func default input. reset() transmit it to env
render_in_train = False  #True #
render_in_demo = False
#2.keep
#set at game.play_games(). refer it at loop in deal_card()
keep_batch_same_in_demo = False
#set at env() creation. impact card shuffle() in reset(). keep constant seed just before full_poker.init() and shuffle()
keep_env_same_in_demo = False
#3.print cycle
#set at game.play_games(). print game result cycle
print_game_result_cycle = 2048

class_short_name = { 'MC_q'            : agent_1.PlayAgentDNN_MC_q,
                     'MC_q_b'          : agent_1.PlayAgentDNN_MC_q_Behavior,
                     'MC_q_sm'         : agent_1.PlayAgentDNN_MC_q_Top3_Softmax,
                     'MC_q_avg'        : agent_1.PlayAgentDNN_MC_q_Top3_Uniform,
                     'MC_q_gd'         : agent_1.PlayAgentDNN_MC_q_guess,
                     'MC_q_gd_b'       : agent_1.PlayAgentDNN_MC_q_guess_Behavior,
                     'MC_q_f'          : agent_1.PlayAgentDNN_MC_q_fraud,
                     'MC_q_f_b'        : agent_1.PlayAgentDNN_MC_q_Behavior_fraud,
                     'MC_pi'           : agent_2.PlayAgentDNN_MC_pi,
                     'MC_pi_b'         : agent_2.PlayAgentDNN_MC_pi_Behavior,
                     'MC_pi_acc'       : agent_2.PlayAgentDNN_MC_pi_aac,
                     'MC_pi_acc_b'     : agent_2.PlayAgentDNN_MC_pi_aac_Behavior,
                     'MC_pi_acc_e'     : agent_2.PlayAgentDNN_MC_pi_aac_elibility,
                     'MC_pi_f'         : agent_2.PlayAgentDNN_MC_pi_fraud,
                     'MC_pi_f_b'       : agent_2.PlayAgentDNN_MC_pi_Behavior_fraud,
                     'MC_pi_acc_f'     : agent_2.PlayAgentDNN_MC_pi_aac_fraud,
                     'MC_pi_acc_f_b'   : agent_2.PlayAgentDNN_MC_pi_aac_Behavior_fraud,
                     'MC_pi_acc_e_f'   : agent_2.PlayAgentDNN_MC_pi_aac_elibility_fraud,
                     'TD_q'            : agent_3.PlayAgentDNN_TD_Expected_q,
                     'TD_q_b'          : agent_3.PlayAgentDNN_TD_Expected_q_Behavior,
                     'TD_q_f'          : agent_3.PlayAgentDNN_TD_Expected_q_fraud,
                     'TD_q_f_b'        : agent_3.PlayAgentDNN_TD_Expected_q_Behavior_fraud,
                     'C_MC_q'          : agent_6.PlayAgentCNN_MC_q,
                     'C_MC_q_b'        : agent_6.PlayAgentCNN_MC_q_Behavior, 
                     'C_MC_q_gc'       : agent_6.PlayAgentCNN_MC_q_guess,
                     'C_MC_q_gc_b'     : agent_6.PlayAgentCNN_MC_q_guess_Behavior, 
                     'Res_MC_q'        : agent_6.PlayAgentRes_MC_q,
                     'Res_MC_q_b'      : agent_6.PlayAgentRes_MC_q_Behavior, 
                     'C_MC_q_f'        : agent_6.PlayAgentCNN_MC_q_fraud,
                     'C_MC_q_f_b'      : agent_6.PlayAgentCNN_MC_q_Behavior_fraud,
                     'Res_MC_pi'       : agent_7.PlayAgentRes_MC_pi,
                     'Res_MC_pi_b'     : agent_7.PlayAgentRes_MC_pi_Behavior,
                     'C_MC_pi'         : agent_7.PlayAgentCNN_MC_pi,
                     'C_MC_pi_b'       : agent_7.PlayAgentCNN_MC_pi_Behavior,
                     'Res_MC_pi_acc'   : agent_7.PlayAgentRes_MC_pi_acc,
                     'Res_MC_pi_acc_b' : agent_7.PlayAgentRes_MC_pi_acc_Behavior,
                     'Res_MC_pi_f'     : agent_7.PlayAgentRes_MC_pi_fraud,
                     'Res_MC_pi_f_b'   : agent_7.PlayAgentRes_MC_pi_Behavior_fraud,
                     'C_TD_q'          : agent_11.PlayAgentCNN_TD_Expected_q,
                     'C_TD_q_b'        : agent_11.PlayAgentCNN_TD_Expected_q_Behavior,
                     }

id_game_id = 0
id_game_env_id     = id_game_id+1
id_game_agent_s    = id_game_env_id+1
id_game_agent_e    = id_game_agent_s+1
id_game_agent_n    = id_game_agent_e+1
id_game_agent_w    = id_game_agent_n+1
id_game_batch      = id_game_agent_w+1
id_game_games      = id_game_batch+1
id_game_keep_batch = id_game_games+1
id_game_demos      = id_game_keep_batch+1
id_game_flag_4in1  = id_game_demos+1

id_env_id = 0
id_env_play_reward_format = id_env_id+1
id_env_keep_env           = id_env_play_reward_format+1

id_agent_id = 0
id_agent_name       = id_agent_id+1
id_agent_net_conf   = id_agent_name+1
id_agent_lr         = id_agent_net_conf+1
id_agent_epsilon    = id_agent_lr+1
id_agent_gamma      = id_agent_epsilon+1


class InputParamters():
    def __init__(self):
        self.game_params    = cfg.game_config_sets
        self.env_params     = cfg.env_config_sets
        self.agent_params   = cfg.agent_config_sets
        self.discard_params = 0  #/TBD
        #self.read_params(2)  #TBD: tu share same global var name among multiple processess. the class MUST be "inited DATA" rather than a "run-time assigned BSS"
        return
    
    def read_params(self, selected_p_set_game):
        #############################
        # game parameters
        #############################
        game_parameters = self.game_params[selected_p_set_game]
        print("read param: game: ", selected_p_set_game, game_parameters)
        self.game_id = game_parameters[id_game_id]
        self.env_id = game_parameters[id_game_env_id]
        self.agent_class_s = game_parameters[id_game_agent_s] #class_short_name[]
        self.agent_class_e = game_parameters[id_game_agent_e] #class_short_name[game_parameters[id_game_agent_e]]
        self.agent_class_n = game_parameters[id_game_agent_n] #class_short_name[game_parameters[id_game_agent_n]]
        self.agent_class_w = game_parameters[id_game_agent_w] #class_short_name[game_parameters[id_game_agent_w]]
        self.batch_size = game_parameters[id_game_batch]
        self.games = game_parameters[id_game_games]
        self.keep_batch_same_in_train = game_parameters[id_game_keep_batch]  #True=one-env keep no change in many episodes. diff-env would be depends on param:keep_env_same_in_train
        self.demos = game_parameters[id_game_demos]
        self.for_in_one = game_parameters[id_game_flag_4in1]
    
        #############################
        # env parameters
        #############################
        selected_p_set_env = -1
        for i, params in enumerate(self.env_params):
            if params[id_env_id] == self.env_id:  #id
                selected_p_set_env = i
                break   #i
        if -1 == selected_p_set_env:
            print("worng config param: env id ....................... ", self.env_id)
            return False

        env_parameters = self.env_params[selected_p_set_env]
        print("read param: env: ", selected_p_set_env, env_parameters)
        self.play_reward_template = env_parameters[id_env_play_reward_format]
        self.keep_env_same_in_train = env_parameters[id_env_keep_env] #False=all env have same card set. used in single card deal batching
    
    
        #############################
        # play agent parameters
        #############################
        self.agent_id = []
        self.agent_class = []
        self.net_conf = []
        self.lr = []
        self.epsilon = []
        self.gamma0 = []
        self.play_file_name_e = []
        self.play_file_name_t = []

        if True == self.for_in_one:
            self.agent_class_e = self.agent_class_s
            self.agent_class_n = self.agent_class_s
            self.agent_class_w = self.agent_class_s
        
        for play_agent_id in [self.agent_class_s, self.agent_class_e, self.agent_class_n, self.agent_class_w]:
            selected_p_set_agent = -1
            for i, params in enumerate(self.agent_params):
                if params[id_agent_id] == play_agent_id:  #id
                    selected_p_set_agent = i
                    break   #i
            if -1 == selected_p_set_agent:
                print("worng config param: agent id ........................... ", play_agent_id)
                return False
            
            agent_parameters = self.agent_params[selected_p_set_agent]
            print("read param: agent: ", selected_p_set_agent, agent_parameters)
            self.agent_id.append(play_agent_id)
            self.agent_class.append(class_short_name[agent_parameters[id_agent_name]])
            self.net_conf.append(agent_parameters[id_agent_net_conf]) #TBD: in multi-processes in UNIX! now, it is wrong!!
            self.lr.append(agent_parameters[id_agent_lr])
            self.epsilon.append(agent_parameters[id_agent_epsilon])
            self.gamma0.append(agent_parameters[id_agent_gamma])
            #self.play_file_name_e.append('./results/play_e_' + str(self.game_id) + '_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')
            #self.play_file_name_t.append('./results/play_t_' + str(self.game_id) + '_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')
            self.play_file_name_e.append('./results/play_e_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')
            self.play_file_name_t.append('./results/play_t_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')

        return True


param_set = InputParamters()
tt0 = time.localtime() #time for training
tt1 = time.localtime()
ydl_cnt = 0
def init_training(reload0, seed0_per_cpu, render, selected_p_set_game, net0=0, save_agent=True):
    #init training
    global tt0, tt1, ydl_cnt

    #isgc = gc.isenabled() 
    #threshold = gc.get_threshold() 
    #print("init thred..............................: ", threshold, isgc)
    #gc.set_threshold(int(1e+6), int(1e+6), int(1e+6))
    #threshold = gc.get_threshold()     
    #print("init thred..............................: ", threshold)

    #load all config parameters again. in sub-process, it is needed. sub-process can't reuse same global var name, whose value is updated in main process in realtime rather than 'initial' moment
    if False == param_set.read_params(selected_p_set_game):
        return 0,0,0,0,0

    if 0 == param_set.games:
        return 0,0,0,0,0
    residual = False
    
    #############################
    # play agents saving
    #############################
    hidden_layers0 = param_set.net_conf[0]
    if 1002 == param_set.agent_class_s:
        hidden_layers = hidden_layers0
    elif 3902 == param_set.agent_class_s:
        hidden_layers = hidden_layers0[0]
        hidden_layers_g = hidden_layers0[1]
    else:
        print("YDL not support ", param_set.agent_class_s)
        return 0,0,0,0,0
    
    learning_rate = param_set.lr[0]
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
    # net=q(s,a)
    ###############
    input_shape0 = (5, 54, 1)
    output_shape0 = 54
    output_shape0_g = 162
    input_shape1 = (4, 55, 1)
    output_shape1 = 55
    input_shape2 = (3, 56, 1)
    output_shape2 = 56
    input_shape3 = (2, 57, 1)
    output_shape3 = 57
    qe_net = CNN_net.build_CNN_network_mem(input_shape0, 
                                        conv_filters_in, kernal_sizes_in, strides_in, 
                                        conv_filter_out, kernal_size_out, stride_out, output_shape0,
                                        residual_net_config, learning_rate)

    '''
    qt_net = CNN_net.build_CNN_network_mem(input_shape1, 
                                        conv_filters_in, kernal_sizes_in, strides_in, 
                                        conv_filter_out, kernal_size_out, stride_out, output_shape1,
                                        residual_net_config, learning_rate)
    
    qn_net = CNN_net.build_CNN_network_mem(input_shape2, 
                                        conv_filters_in, kernal_sizes_in, strides_in, 
                                        conv_filter_out, kernal_size_out, stride_out,output_shape2,
                                        residual_net_config, learning_rate)
    qm_net = CNN_net.build_CNN_network_mem(input_shape3, 
                                        conv_filters_in, kernal_sizes_in, strides_in, 
                                        conv_filter_out, kernal_size_out, stride_out, output_shape3,
                                        residual_net_config, learning_rate)
    '''

    if 3902 == param_set.agent_class_s:
        qe_guess = guess.PlayGuess_CNN(hidden_layers=hidden_layers_g, filename_g='qe_guess', learning_rate=0.01, reload=False, learning_amount=10)
        '''
        qt_guess = guess.PlayGuess_CNN(hidden_layers=hidden_layers[1], filename_g='qt_guess', learning_rate=0.01, reload=False, learning_amount=10)
        qn_guess = guess.PlayGuess_CNN(hidden_layers=hidden_layers[1], filename_g='qn_guess', learning_rate=0.01, reload=False, learning_amount=10)
        qn_guess = guess.PlayGuess_CNN(hidden_layers=hidden_layers[1], filename_g='qm_guess', learning_rate=0.01, reload=False, learning_amount=10)
        '''

    fit_batch = 256
    batch_size = param_set.batch_size * 11 * 4 #11 round in a game and 4in1
    ydl0 = np.zeros((batch_size, output_shape0))
    ydl1 = np.zeros((batch_size, output_shape1))
    ydl2 = np.zeros((batch_size, output_shape2))
    ydl3 = np.zeros((batch_size, output_shape3))
    ydl0_g = np.zeros((batch_size, output_shape0_g))
    
    print("TRAINing start ..................................................", time.strftime("%Y-%m-%d_%H-%M-%S", tt0))
    for i in range(0, param_set.games, param_set.batch_size):
        x_data = np.random.randn(batch_size, input_shape0[0], input_shape0[1], input_shape0[2]) 
        ydl0 = qe_net.predict(x_data, batch_size=fit_batch)
        y_data = np.random.randn(batch_size, output_shape0) + ydl0
        qe_net.fit(x_data, y_data, verbose=0, batch_size=fit_batch)
        ydl_cnt += 1
        if ydl_cnt % 100 == 0 :
            print("qe_net    X, Y shape ", ydl_cnt, x_data.shape, y_data.shape, fit_batch)
        #qt_net.set_weights(qe_net.get_weights())
        '''
        x_data = np.random.randn(batch_size, input_shape1[0], input_shape1[1], input_shape1[2]) 
        ydl1 = qt_net.predict(x_data, batch_size=fit_batch)
        y_data = np.random.randn(batch_size, output_shape1) + ydl1
        qt_net.fit(x_data, y_data, verbose=0, batch_size=fit_batch)
    
        x_data = np.random.randn(batch_size, input_shape2[0], input_shape2[1], input_shape2[2]) 
        ydl2 = qn_net.predict(x_data, batch_size=fit_batch)
        y_data = np.random.randn(batch_size, output_shape2) + ydl2
        qn_net.fit(x_data, y_data, verbose=0, batch_size=fit_batch)

        x_data = np.random.randn(batch_size, input_shape3[0], input_shape3[1], input_shape3[2]) 
        ydl3 = qm_net.predict(x_data, batch_size=fit_batch)
        y_data = np.random.randn(batch_size, output_shape3) + ydl3
        qm_net.fit(x_data, y_data, verbose=0, batch_size=fit_batch)
        '''
        if 3902 == param_set.agent_class_s:
            x_data = np.random.randn(batch_size, input_shape0[0], input_shape0[1], input_shape0[2]) 
            ydl0_g = qe_guess.guess_net.predict(x_data, batch_size=fit_batch)
            y_data = np.random.randn(batch_size, output_shape0_g) + ydl0_g
            qe_guess.guess_net.fit(x_data, y_data, verbose=0, batch_size=fit_batch)
            if ydl_cnt % 100 == 0:
                print("guess_net X, Y shape ", ydl_cnt, x_data.shape, y_data.shape, fit_batch)
            
        time.sleep(0.5)
        
    tt1 = time.localtime()
    print("TRAINing finsih..................................................", time.strftime("%Y-%m-%d_%H-%M-%S", tt1))


    return ydl0, ydl1, ydl2, ydl3, ydl0_g

    
def main(argv):
    print("argv ", argv)
#-r init -m 3 -c 7 -s 13 -p 1
#-r init -p 4
#-r init -g 0 -p 4                        #only GPU=0. 'games'*10 if -g
#-r init -m 3 -c 7 -s 13 -t 50 -p 1000    #from p 1000, run 50 configs continuous, on CPU 5,6,7. #can't strictly stick a 'process' to a 'CPU'(in Debian. CentOS works). replaced by 'many_taskset2.py'
#-r resume -m 3 -c 7 -s 13 -p 0
#-r resume -m 3 -c 7 -s 13 -t 50 -p 0     #from p100000, run 50 configs, on CPU 5,6,7
#-r comp -m 3 -c 7 -a 50 -p 100000  #from p100000, run 50 configs, on CPU 5,6,7
#-r test -p 800
#-r test -u 1 -p 800  #massive
#-r test -u 2 -p 800  #auto
#-r test -u 3 -p 824  #sanity: training() for configs [810, selected_p_set3]
    multi_proces = 0
    cpu_back_start = 7
    seed_start = 13
    net0_list1 = 0
    selected_p_set = []
    all_competitions = 0
    perform_UT = False
    test_auto_level = 0
    selected_p_set3 = 0
    enable_GPU = False
    from_to = 0

    ####################
    # get command line input params
    ####################
    try:
        opts, args = getopt.getopt(argv,"r:m:c:s:p:a:u:t:")
    except getopt.GetoptError:
        print("wrong input")
        return;

    try:
        for opt, arg in opts:
            print("arg: ",opt, arg)
            if opt == '-r':
                if arg == 'init' :
                    reload = False
                elif arg == 'resume' :
                    reload = True
                elif arg == 'comp' :
                    reload = True
                elif arg == 'test' :
                    reload = False
                    perform_UT = True
                elif opt == '-g':  #actually, -g doesn't work. enable/disable the GPU, have to do so before main during imported pkgs start up
                    gpu_id = int(arg)
                    GPU_prepare(gpu_id)  #only GPU-0 is supported
                    enable_GPU = True
                else:
                    print("wrong -r input", opt, arg)
                    return

            if opt == '-m': #multi process: 0,1-27
                multi_proces = int(arg)

            if opt == '-c': #7 or 27
                cpu_back_start = int(arg)

            if opt == '-s':
                seed_start = int(arg)
            
            if opt == '-a':
                all_competitions = int(arg)

            if opt == '-u':
                test_auto_level = int(arg)

            if opt == '-t':
                from_to = int(arg)

            if opt == '-p':
                selected_p_set0 = arg.split(',')
                selected_p_set1 = [int(c) for c in selected_p_set0] #support only 1 param set
                selected_p_set = selected_p_set1[0]
                for i, params in enumerate(cfg.game_config_sets):
                    if params[id_game_id] == selected_p_set:  #id
                        selected_p_set2 = i
                        break   #i
                if True == perform_UT and 3 == test_auto_level:
                    #don't search the line ID when sanity
                    selected_p_set3 = selected_p_set
                
    except  ValueError:
            print("wrong input", opt, arg)
            return

    print("input set: multi-porcess + cpu start + seed start + param set + param set id + all games: ", multi_proces, cpu_back_start, seed_start, selected_p_set, selected_p_set2, all_competitions)

    #load all config parameters into main processor, which can't be propagated to sub-process
    if False == param_set.read_params(selected_p_set2):
        return
    
    seed = seed_start 
    _, _, _, _, _ = init_training(reload, seed, render_in_train, selected_p_set2)

    return


if __name__ == "__main__":
    t0 = time.localtime()
   
    ######################################
    # temp test field
    ######################################
    class base:
        def __init__(self):
            print('in base')
            print('out base')
            
    class A(base):
        def __init__(self):
            print('in A')
            super(A, self).__init__()
            print('out A')


    class B(A):
        def __init__(self):
            print('in B')
            super(B, self).__init__()
            print('out B')


    class C(B, A):
        def __init__(self):
            print('in C')
            super(C, self).__init__()
            print('out C')

    ydl = C()

    #os.popen('rm ./results/tmp.txt') #doesn't work in windows
    a1 = np.arange(110).reshape(11,2,5)
    a2 = a1.swapaxes(1, 2)
    a3 = a1.reshape(-1, 2)
    

    def zip_i(n):
        # zip loop is faster than i loop. 差距主要来自于对i索引的开销。
        # zip 本身比 range(n)耗时， 但处理中的i寻址的消耗的绝对时间更大。总体看，zip快，但不好量化
        # (inline loop + zip) is faster than (outline loop + zip)
        # (inline loop + range(i) is slower than (outline loop + range(i)) !!
        m = 100
        ydl1 = np.arange(n*m).reshape(-1, m)
        ydl2 = np.arange(1,2*n*m+1,2).reshape(-1, m)
        
        tc11 = time.time()
        ydl3 = [ydl1[i]+ydl2[i] for i in range(n)]
        tc12 = time.time()
        perf1 = tc12 - tc11

        tc31 = time.time()
        for i in range(n):
            ydl3 = ydl1[i]+ydl2[i]
        tc32 = time.time()
        perf3 = tc32 - tc31

        tc21 = time.time()
        ydl4 = [ydl11+ydl21 for ydl11, ydl21 in zip(ydl1, ydl2)]
        tc22 = time.time()
        perf2 = tc22 - tc21

        tc41 = time.time()
        for ydl11, ydl21 in zip(ydl1, ydl2):
            ydl4 = ydl11+ydl21
        tc42 = time.time()
        perf4 = tc42 - tc41

        perf5 = 0
        for i in range(n):
            tc51 = time.time()
            ydl4 = ydl1[i]+ydl2[i]
            tc52 = time.time()
            perf5 += tc52 - tc51
        
        perf6 = 0
        for ydl11, ydl21 in zip(ydl1, ydl2):
            tc61 = time.time()
            #ydl3 = ydl0 + ydl00
            ydl4 = ydl11 + ydl21
            tc62 = time.time()
            perf6 += tc62 - tc61

        return perf1, perf2, perf3, perf4, perf5, perf6
        
    #perf1, perf2, perf3, perf4, perf5, perf6 = zip_i(int(5e+6))
    #print(perf1, perf2, perf3, perf4, perf5, perf6)
    
    
    a1 = np.array([ 4, 5, 6, 1, 2, 3, 7, 86, 9, 15])
    a2 = np.array([ 2, 3, 5, 6, 9])
    ydl = np.s_[0:5:2]
    ydl2 = a1[ydl]

    '''
    import tensorflow.compat.v2 as tf
    from tensorflow import keras

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=2, input_shape=(2,), activation=tf.nn.relu)) #(input_size[0],input_size[1])
    model.add(keras.layers.Dense(units=1, activation=None)) # 输出层
    optimizer = tf.optimizers.Adam(lr=0.001)
    model.compile(loss=tf.losses.mse, optimizer=optimizer) #, metrics=metrics) #['accuracy']), his['acc']
    model.summary()
    ydl = model.get_weights()
    
    #meas.get_size(model)

    session = tf.Session()
    keras.backend.set_session(session)
    keras.backend.clear_session()
    tf.reset_default_graph()
    '''
    # temp test field

    ######################################
    # offical starting ...
    ######################################
    main(sys.argv[1:])
    print("YDL ", time.strftime("%Y-%m-%d_%H-%M-%S", t0))
    print("YDL ", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    print("YDL train", time.strftime("%Y-%m-%d_%H-%M-%S", tt0))
    print("YDL train", time.strftime("%Y-%m-%d_%H-%M-%S", tt1))
