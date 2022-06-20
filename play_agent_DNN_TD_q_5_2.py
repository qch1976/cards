import time
import numpy as np
import tensorflow.compat.v2 as tf2

import play_agent_DNN_MC_q_5_2 as DNN_q_base
import game_verify_online_5_2 as game_verify
from traj_replayer_5_2 import TrajectoryReplayer
'''
class TrajectoryReplayer:
    def __init__(self, capacity):
        #the mempory size of state3s is huge, if multiple CPUs, such as 10 CPUs parallel.
        #one game size = (1, 11, 4, 1, 5, 54)*2bytes(float16)*(batch_size=1)=23.76k
        #200k games(regardless of batch)=4.752G, *10CPU = 47.52G !!!
        #TBD: reference to: memory realtime data compress: https://sparse.pydata.org/en/latest/quickstart.html
        #TBD: spares packet install failure!!!
        self.np_states  = np.array([], dtype=np.float16)
        self.np_actions = np.array([])
        self.np_rewards = np.array([])
        self.np_bs      = np.array([])
        self.np_next_states = np.array([])
        self.np_priorities  = np.array([])

        self.upper_mem_limitation = capacity
        return
    
    def store(self, np_states, np_actions, np_rewards, np_bs, np_next_states):
        #np_state3s and np_next_states shape=(11-player*batch-info(5-54), or(4-54*2))
        #np_actions, np_rewards shape=(11-player*batch-info(0))
        #bs shape=(11-player*batch-info(0))
        np_actions = np_actions.reshape(-1)
        np_rewards = np_rewards.reshape(-1)
        np_bs = np_bs.reshape(-1)
        state_shape = np_states.shape
        np_states = np_states.reshape(-1, state_shape[2], state_shape[3])
        np_next_states = np_next_states.reshape(-1, state_shape[2], state_shape[3])
        ### verify: 
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_13, np_next_states)
        
        np_priorities = self.prioritize(np_states, np_rewards)
        #setup a upper limitation of memory
        #np.delete() and np.concatenate() will allocate same size memory of object during operation since they are not taken "in-place'
        #that means during start(), the peak memory size would be doubled !!
        #TBD: may be 'list' has better memory performance then numpy
        memory_size = self.np_states.nbytes + self.np_next_states.nbytes
        #print("store bytes: ", memory_size)
        if memory_size >= self.upper_mem_limitation:
            #remove oldest data and add the latest one later
            obsolete_len = state_shape[0] * 0.3
            self.np_states  = np.delete(self.np_states, np.s_[0:obsolete_len], axis=0)
            self.np_actions = np.delete(self.np_actions, np.s_[0:obsolete_len], 0, axis=0)
            self.np_rewards = np.delete(self.np_rewards, np.s_[0:obsolete_len], 0, axis=0)
            self.np_bs      = np.delete(self.np_bs, np.s_[0:obsolete_len], 0, axis=0)
            self.np_next_states = np.delete(self.np_next_states, np.s_[0:obsolete_len], 0, axis=0)
            self.np_priorities = np.delete(self.np_priorities, np.s_[0:obsolete_len], 0, axis=0)
            
        #self.xxx shape: game-round-player*bacth-info
        #np_actions = np_actions[np.newaxis,:]
        #np_rewards = np_rewards[np.newaxis,:]
        if self.np_states.shape[0] == 0:
            self.np_states  = np_states.astype(np.float16)
            self.np_actions = np_actions.astype(np.int8)
            self.np_rewards = np_rewards.astype(np.float16)
            self.np_bs      = np_bs.astype(np.float16)
            self.np_next_states = np_next_states.astype(np.float16)
            self.np_priorities  = np_priorities.astype(np.int8)
        else:
            self.np_states  = np.concatenate((self.np_states, np_states.astype(np.float16)), axis=0)
            self.np_actions = np.concatenate((self.np_actions, np_actions.astype(np.int8)), axis=0)
            self.np_rewards = np.concatenate((self.np_rewards, np_rewards.astype(np.float16)), axis=0)
            self.np_bs      = np.concatenate((self.np_bs, np_bs.astype(np.float16)), axis=0)
            self.np_next_states = np.concatenate((self.np_next_states, np_next_states.astype(np.float16)), axis=0)
            self.np_priorities  = np.concatenate((self.np_priorities, np_priorities.astype(np.int8)), axis=0)
    
    def sample(self, volume):
        length = self.np_states.shape[0]
        #sample volume from storage
        candidates = min(volume, length)
        #TBD: priority
        selected_trajectory = np.random.choice(length, size=candidates, replace=False)
        return self.np_states[selected_trajectory], self.np_actions[selected_trajectory], self.np_rewards[selected_trajectory], self.np_bs[selected_trajectory], self.np_next_states[selected_trajectory]
    
    def prioritize(self, np_state3s, np_rewards):
        #what is the priority expeirence?
        np_pri_temp = np.full(np_rewards.shape, 1, dtype=np.int8)
        return np_pri_temp
'''
ydl_cnt = 0
class PlayAgentDNN_TD_Expected_q(DNN_q_base.PlayAgentDNN_MC_q):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'TD'], fraud=5):
        if net0_list != 0:
            print("PlayAgentDNN_TD: inside. ")
        DNN_q_base.PlayAgentDNN_MC_q.__init__(self, hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        self.replayer = TrajectoryReplayer(2e+8, (5,54), (5,54))  #2e+9 = 99G/27CPU=3.7G, 3.7G/2~=1.9G. set the limit to 2G
        self.qt_net_refresh_limit = 5
        self.qt_net_refresh_cnt = 0
        return

    def copy_qt(self):
        self.qt_net.set_weights(self.qe_net.get_weights())

    #no additional processing. easy for debug
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch=state3s_batch, available_mask_batch=available_mask_batch, train=train)

    def learning_multi_games(self):
        self.sampling_learning(1000)
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #input state3, next_states_batch: shape: round(=11)-player*batch-info(5, 54), or (4, 54*2)
        #action, reward shape: round(=11)-player*batch-info(0). no change in this method

        if isinstance(behaviors_batch, int):
            behaviors_batch = np.full(actions_batch.shape, 1, dtype=np.float16)
            
        self.replayer.store(state3s_batch, actions_batch, rewards_batch, behaviors_batch, next_states_batch)
        
        TD_sample_batch = state3s_batch.shape[0] * state3s_batch.shape[1]
        self.sampling_learning(TD_sample_batch) #11*4*100, 100
    
    def shapes(self, learning_states, learning_next_states):
        next_state_shape = (learning_next_states.shape[0], -1)
        state_shape = (learning_states.shape[0], -1)
        return state_shape, next_state_shape
        
    def sampling_learning(self, samples):
        global ydl_cnt
        
        learning_states, learning_actions, learning_rewards, learning_bs, learning_next_states, _ = self.replayer.sample(samples)
        state_shape, next_state_shape = self.shapes(learning_states, learning_next_states)
        
        if 0 == state_shape[0] or 0 == next_state_shape[0]:
            history = []
        else:
            bitmasks_like = np.where(learning_next_states[:,0,:]>0, 1, 0)[:,:54]  #it is not actual bitmask since no info about (1) 'first played card' even the 'first player' is known. (2)'trump', 'round' in env. agent doesn't know any env data
            learning_next_action_n = np.sum(bitmasks_like, axis=1)
            dones = np.where(learning_next_action_n>1, 0 ,1)  #True mean done
    
            learning_next_qs0 = self.qt_net.predict(learning_next_states.reshape(next_state_shape))
            learning_next_qs = learning_next_qs0 * bitmasks_like  #[:54] in order to compabile to fraud
            
            #apply 'expected SARSA' with even distribution... bad idea
            #doesn't use epsilon since it can't decide the Qmax. the actual bitmask info can't be calculated
            next_Vs = np.sum(learning_next_qs, axis=1) / learning_next_action_n  #give every actions even possibility
            Us = learning_rewards + self.gamma * next_Vs * (1. - dones)
            
            q_targets0 = self.qe_net.predict(learning_states.reshape(state_shape))
            bitmasks = np.where(learning_states[:,0,:]>0, 1, 0)[:,:54]  #it is not actual bitmask since no info about (1) 'first played card' even the 'first player' is known. (2)'trump', 'round' in env. agent doesn't know any env data
            q_targets = q_targets0 * bitmasks
            q_targets[np.arange(Us.shape[0]), learning_actions] = Us
    
            self.sync_acquire()
            self.sync_mp_net0_to_local(self.net0_list, self.primay_net)
            history = self.qe_net.fit(learning_states.reshape(state_shape), q_targets, verbose=0, batch_size=256)
            self.sync_local_to_mp_net0(self.primay_net, self.net0_list)
            self.sync_release()

            ydl_cnt += 1
            if ydl_cnt % 100 == 0 :
                print("qe_net    X, Y shape ", ydl_cnt, learning_states.reshape(state_shape).shape, q_targets.shape, 256)
    
            self.qt_net_refresh_cnt += 1
            if self.qt_net_refresh_cnt >= self.qt_net_refresh_limit:
                self.qt_net_refresh_cnt = 0
                self.qt_net.set_weights(self.qe_net.get_weights())  #learn在round过程中不会调用, then if limit=0, 函数里的qt可以直接用qe
    
    
            ''' #有限状态, refernce
            v = (self.q[next_state].sum() * self.epsilon / self.action_n + self.q[next_state].max() * (1. - self.epsilon))
            u = reward + self.gamma * v * (1. - done)
            td_error = u - self.q[state, action]
            self.q[state, action] += self.learning_rate * td_error
            ''' 
    
            ''' #NN reference
            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs.max(axis=-1)  # =>Q
            us = rewards + self.gamma * (1. - dones) * next_max_qs
    
            ###用evaluate model训练，用U更新 predict出的经验数据
            #evaluate model生成state的q(s,)
            #targets=[64,3], actions=[64]
            targets = self.evaluate_net.predict(observations)
            #Q: why？ target model的U替换evaluate model的输出值q(s,*)
            #A: 近似法替代迭代法，用DN时，q(s,a)值不能用 [U - q(s,a)]直接更新，而是用DN train代替
            targets[np.arange(us.shape[0]), actions] = us
    
            #update评估网络的w only.
            #用evaluate model的perdict + U update,训练evaluate model
            self.evaluate_net.fit(observations, targets, verbose=0)
    
            if done: # 更新目标网络
                self.target_net.set_weights(self.evaluate_net.get_weights())
            '''

        return history



class PlayAgentDNN_TD_Expected_q_Behavior(PlayAgentDNN_TD_Expected_q): #, DNN_q_base.PlayAgentDNN_MC_q_Behavior):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'TD'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentDNN_TD_Expected_q_Behavior, train=train)




class PlayAgentDNN_TD_Expected_q_fraud(PlayAgentDNN_TD_Expected_q):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'fraud', 'TD'], fraud=4):
        PlayAgentDNN_TD_Expected_q.__init__(self, hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        self.replayer = TrajectoryReplayer(2e+8, (4,54*2), (4,54*2))  #2e+9 = 99G/27CPU=3.7G, 3.7G/2~=1.9G. set the limit to 2G
        return

    #no additional processing. easy for debug
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch=state3s_batch, available_mask_batch=available_mask_batch, train=train)

    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #input state3, next_states_batch: shape: round(=11)-player*batch-info(5, 54), or (4, 54*2)
        #action, reward shape: round(=11)-player*batch-info(0). no change in this method
        try:
            history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=behaviors_batch)
        except TypeError as e:
            print(e)
            
        return history



class PlayAgentDNN_TD_Expected_q_Behavior_fraud(PlayAgentDNN_TD_Expected_q_fraud):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'fraud', 'TD'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentDNN_TD_Expected_q_Behavior_fraud, train=train)

