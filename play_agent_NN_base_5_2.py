
import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras
import psutil
import copy

import game_verify_online_5_2 as game_verify

ydl_net = 0

ydl_cnt_q = 0
ydl_cnt_pi = 0

class PlayAgentNNBase:
    def __init__(self, learning_rate, epsilon=0.2, gamma=0.0, net0_list=0, name=['non-behavior']):
        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon
        self.net0_list = net0_list
        self.primay_net = 0
        self.pid = psutil.Process().pid  # get self pid
        print("play PID: ", self.pid)
        self.value_err_cnt = 0
        
    ####################################
    # reinforcement learning NON-related common methods
    ####################################
    def get_primay_net(self):
        return self.primay_net
    

    def save_model(self, net, filename):
        global ydl_net
        self.sync_acquire()
        net.save(filename)
        self.sync_release()
        print("save_model: ydl: filename", filename)
        if isinstance(ydl_net, int):
            ydl_net = net.get_weights() #only primary net
        if self.net0_list != 0:
            print("save_model: ydl: ", ydl[0][0,1], ydl[2][50,50], ydl[4][126,2])
            print("save_model: net0 id: ", id(self.net0_list), self.net0_list[0][0][0,1], self.net0_list[0][2][50,50], self.net0_list[0][4][126,2])
        

    def load_model(self, filename, my_loss=0):
        net = keras.models.load_model(filename, custom_objects={'my_loss': my_loss}) #{'ydl_loss': ydl_loss}, NAME MUST BE SAME
        print("load_model: ydl: filename", filename)
        ydl = net.get_weights()
        if self.net0_list != 0:
            print("load_model: ydl: ", filename, ydl[0][0,1], ydl[2][50,50], ydl[4][126,2])
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
        #print(ydl_t, "play agent base:sync_local_to_mp_net0: net0 before ", self.pid, id(net0_list), 
        #      net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
        #print(ydl_t, "play agent base:sync_local_to_mp_net0: local before ", self.pid, id(net0_list),
        #      net_weights[0][0,1], net_weights[2][50,50], net_weights[4][126,2])

        #verify
        net_local_shape = [weight.shape for weight in net_weights]
        #net_local_np = np.array(net_weights)
        #net_local_shape = [net_local_np[i].shape for i in range(net_local_np.shape[0])]
        #print("sync_local_to_mp_net0 get_weights: ", net_local_shape)
        #print("sync_local_to_mp_net0 net0_list shape: ", net0_list[2])
        
        if net_local_shape == net0_list[2]:
            #net0_list[1].acquire()
            net0_list[0] = copy.deepcopy(net_weights) #weight matrix
            #net0_list[1].release()
            
            #verify
            ydl_t = time.time()
            #print(time.time(), "play agent base:sync_local_to_mp_net0: net0 after ", self.pid, id(net0_list),
            #      net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
            #print(ydl_t, "play agent base:sync_local_to_mp_net0: local after ", self.pid, id(net0_list),
            #      net_weights[0][0,1], net_weights[2][50,50], net_weights[4][126,2])
        else:
            print("play agent base:sync_local_to_mp_net0:after , shape wrong", self.pid, id(net0_list))
        
        return

    def sync_mp_net0_to_local(self, net0_list, net):
        if net0_list == 0:
            return
        
        net_weights = net.get_weights()
        #print("agent6 base:sync_mp_net0_to_local:id ", id(self.net0_list))
        ydl_t = time.time()

        #print(ydl_t, "play agent base:sync_mp_net0_to_local: net0 before ", self.pid, id(net0_list), 
        #      net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
        #print(ydl_t, "play agent base:sync_mp_net0_to_local: local before ", self.pid, id(net0_list), 
        #      net_weights[0][0,1], net_weights[2][50,50], net_weights[4][126,2])

        #verify
        net_local_shape = [weight.shape for weight in net_weights]
        #net_local_np = np.array(net_weights)
        #net_local_shape = [net_local_np[i].shape for i in range(len(net_local_np))]
        #print("sync_mp_net0_to_local get_weights: ", net_local_shape)

        #net0_list_np = [np.array(net0_list[0][i]) for i in range(len(net0_list[0]))]

        if net_local_shape == net0_list[2]:
            #net0_list[1].acquire()
            net.set_weights(net0_list[0]) # net0_list_np
            #net0_list[1].release()
        
            #verify
            net_weights = net.get_weights()
            ydl_t = time.time()
            #print(ydl_t, "play agent base:sync_mp_net0_to_local: net0 after ", self.pid, id(net0_list), 
            #      net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
            #print(ydl_t, "play agent base:sync_mp_net0_to_local:local after ", self.pid, id(net0_list),
            #      net_weights[0][0,1], net_weights[2][50,50], net_weights[4][126,2])
        else:
            print("play agent base:sync_mp_net0_to_local:after , shape wrong", self.pid, id(net0_list))

    def sync_acquire(self):
        if self.net0_list == 0:
            return
        else:
            self.net0_list[1].acquire()

    def sync_release(self):
        if self.net0_list == 0:
            return
        else:
            self.net0_list[1].release()

    def first_net0_sync(self):
        if self.net0_list != 0 and self.primay_net != 0:
            print("PlayAgentNN_base: net0 id1 + init: ", id(self.net0_list), self.net0_list[0][0][0,1], self.net0_list[0][2][50,50], self.net0_list[0][4][126,2])

            if (7 == self.net0_list[0][0][0,1] and 3 == self.net0_list[0][2][50,50] and 7 == self.net0_list[0][4][126,2]):
                #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                print("PlayAgentNN_base: first copy to net0")
                self.sync_acquire()
                self.sync_local_to_mp_net0(self.primay_net, self.net0_list)
                self.sync_release()
            else:
                print("PlayAgentNN_base: second+ copy to local")
                self.sync_acquire()
                self.sync_mp_net0_to_local(self.net0_list, self.primay_net)
                self.sync_release()
        else:
            print("PlayAgentNN_base: net0_list or primay_net is empty !")
            

    ####################################
    # reinforcement learning RELATED common methods decouple from specific agent
    ####################################
    def decide_54_q(self, state3s_batch, available_mask_batch):
        # normal
        targets0 = self.primay_net.predict(state3s_batch) #.reshape(batch_size,-1); backgroud with previous predict
        #targets1 = targets0 * available_mask_batch  #clear the position that oindex not existing
        #targets = np.where(targets1==0, float("-inf"), targets1) #to avoid the case: 0 is the max
        
        #replace above 2 lines
        targets = np.where(available_mask_batch==0, float("-inf"), targets0) #clear the position that oindex not existing. and to avoid the case: 0 is the max
        return targets, targets0  #shape=(n, 54)

    def decide_q(self, state3s_batch, available_mask_batch, train=True):
        batch_size = state3s_batch.shape[0]
        oindex = []
        
        ydl = np.random.uniform(0.0, 1.0, (1,))
        if  ydl < self.epsilon and train==True:
            for i in range(batch_size):
                oindex0 = np.where(available_mask_batch[i]==True)[0]
                oindex.append(np.random.choice(oindex0))
            action_oindex = np.array(oindex)
        else:
            targets, targets0 = self.decide_54_q(state3s_batch, available_mask_batch)
            action_oindex = np.argmax(targets, axis=1)
            #print("decide: ", action_oindex, targets0[:,action_oindex])
            
            actions0_sorted_oindex = np.argsort(-targets0)  #(-):bigger -> smaller
            actions0_max_oindexes = actions0_sorted_oindex[:, 0:6]
            actions_sorted_oindex = np.argsort(-targets)  #(-):bigger -> smaller
            actions_max_oindexes = actions_sorted_oindex[:, 0:3]
            
        return action_oindex #shape=(n, 54)

    def decide_pi(self, state3s_batch, available_mask_batch, train=True, piv_net=False):
        candidates = 3  #select based on top-n possibility. TBD: diff to epsilon ?? q() based policy, uses epsilon??
        batch_size = state3s_batch.shape[0]

        if True == piv_net:
            targets_pi0, targets_v = self.primay_net.predict(state3s_batch) #.reshape(batch_size,-1); backgroud with previous predict
        else:
            targets_v = self.secondary_net.predict(state3s_batch)  #v_net
            targets_pi0 = self.primay_net.predict(state3s_batch) #.reshape(batch_size,-1); backgroud with previous predict
        #targets_pi = targets_pi0 * available_mask_batch  #clear the position that oindex not existing
        #replace above line
        targets_pi = np.where(available_mask_batch==0, float("-inf"), targets_pi0) #clear the position that oindex not existing. and to avoid the case: 0 is the max

        actions_sorted_oindex = np.argsort(-targets_pi, axis=1)  #(-):bigger -> smaller
        actions_max_oindex = actions_sorted_oindex[:,0:candidates]  # regardless 3 > available_len or not. =0
        
        batch_n0 = np.arange(0,batch_size).reshape(1,-1)  #batch_size
        batch_n = np.repeat(batch_n0, candidates, axis=0).T #max available len
        
        actions_max_targets0 = targets_pi[batch_n.reshape(-1), actions_max_oindex.reshape(-1)]
        actions_max_targets1 = actions_max_targets0.reshape(batch_size, -1)

        #re-calculate in order to sum(possibility)=100%
        action_oindex = []
        actions_max_targets = actions_max_targets1/np.sum(actions_max_targets1, axis=1)[:,np.newaxis]
        for i, (oindex, p_action) in enumerate(zip(actions_max_oindex, actions_max_targets)):
            try:
                action_oindex0 = np.random.choice(oindex, p=p_action)
                action_oindex.append(action_oindex0)
            except ValueError: 
                if 0 == self.value_err_cnt % 4096:
                    print("decide_pi: probabilities contain NaN. all target/possib are 0", self.value_err_cnt, "p-action: ", p_action, oindex)
                self.value_err_cnt += 1
                
                temp_oindex = np.where(available_mask_batch[i]==1)[0]
                action_oindex0 = np.random.choice(temp_oindex) #evenly choice
                action_oindex.append(action_oindex0)
        return np.array(action_oindex)



    def decide_b(self, state3s_batch, available_mask_batch):
        b = []
        oindex = []
        
        for mask in available_mask_batch:
            oindex0 = np.where(mask==True)[0]
            oindex.append(np.random.choice(oindex0))
            b.append(1.0/len(oindex0))

        action_oindex = np.array(oindex)
        action_b = np.array(b)
        return action_oindex, action_b

    def decide_b_pre(self, state3s_batch, available_mask_batch, super_pre, train=True):
        batch_size = state3s_batch.shape[0]
        
        if True == train:
            action_oindex, action_b = self.decide_b(state3s_batch, available_mask_batch)
        else:
            #super_pre指定调用类，super()计算出父类
            action_oindex = super(super_pre, self).decide(state3s_batch, available_mask_batch, train=False)
            action_b = np.array([1]*batch_size)
        return action_oindex, action_b

    def learning_single_game_q(self, state3s_batch, actions_batch, rewards_batch, bitmasks54, behaviors_batch=0):  #don't know how to add b in NN_q net
        global ydl_cnt_q
        
        #xxx_q can't recog the network type. the shape of state3s_batch must be corectly reshape outside the method!!
        #state3 shape: round(=11)*player*batch-info(network specific shape)
        # state3 in DNN info shape=(5*54), or (4*54)
        # state3 in CNN info shape=(5,54,1), or (4,54,1)
        #action, reward shape: round(=11)-player*batch-info(0)
        #batch_size  = round*player*batch
        #batch_size2 = player*batch
        batch_size = state3s_batch.shape[0]
        batch_size2 = actions_batch.shape[1]
        
        Gs = []
        G = np.array([[0]*batch_size2])
        for rewards in reversed(rewards_batch): #loop: round: 0-10
            G =  rewards + self.gamma * G
            Gs.insert(0,G)

        #reshape round(=11)-player*batch-info(1) to round*player*batch
        actions_batch2 = np.array(actions_batch).flatten('C')  #default = order(C)
        Gs_batch = np.array(Gs).flatten('C')

        #to build the mask of 'card in hand'
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud, 54)  #common shape in DNN and CNN

        ### verify. unknow network type. verify has to detect the type and shape
        #_ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_11, state3s_batch, state3s_batch2, Gs_batch, Gs, actions_batch2, actions_batch)
        
        #should be here: keep other action's q
        #state3s_1 = np.where(state3s_batch2[:,0,:]>0, 1, 0)  #>0 mean in-hand or trump
        targets0 = self.primay_net.predict(state3s_batch) #backgroud with previous predict
        targets = targets0 * bitmasks54  #clear the position that oindex not existing
            
        targets[np.arange(batch_size), actions_batch2] = Gs_batch

        #lock outside of the sync_xxx()
        self.sync_acquire()
        self.sync_mp_net0_to_local(self.net0_list, self.primay_net)
        history = self.primay_net.fit(state3s_batch, targets, verbose=0, batch_size=256)
        self.sync_local_to_mp_net0(self.primay_net, self.net0_list)
        self.sync_release()

        ydl_cnt_q += 1
        if ydl_cnt_q % 100 == 0 :
            print("primay_net    X, Y shape ", ydl_cnt_q, state3s_batch.shape, targets.shape, 256)
        
        return history

    def learning_single_game_pi(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0, piv_net=False):
        global ydl_cnt_pi
        
        #xxx_pi can't recog the network type. the shape of state3s_batch must be corectly reshape outside the method!!
        #state3 shape: round(=11)*player*batch-info(network specific shape)
        # state3 in DNN info shape=(5*54), or (4*54)
        # state3 in CNN info shape=(5,54,1), or (4,54,1)
        #action, reward shape: round(=11)-player*batch-info(1)
        #batch_size  = round*player*batch
        #batch_size2 = player*batch
        batch_size = state3s_batch.shape[0]
        batch_size2 = actions_batch.shape[1]
        if 'non-behavior' in self.name: #not a offline policy
            behaviors_batch = np.full((11, batch_size2), 1)

        if True == piv_net:
            pis, vs0 = self.primay_net.predict(state3s_batch)
        else:
            vs0 = self.secondary_net.predict(state3s_batch)
        vs = vs0.reshape(11, batch_size2)
        
        T = 11
        gamma_t = self.gamma **(T-1)
        G = np.array([[0]*batch_size2])
        Gs = []
        Gs_gamma = []
        
        for t in reversed(range(T)): #[::-1]: 0-5
            G =  rewards_batch[t] + self.gamma * G
            Gs.insert(0, G)
            
            G_v = G - vs[t]
            G_gamma = G_v * gamma_t
            G_gamma /= behaviors_batch[t]  #cum_behavior  # **t: diff to ch7.py
            Gs_gamma.insert(0, G_gamma)

            gamma_t /= self.gamma
            
        Gs_batch = np.array(Gs).reshape(-1,1)  #[:, np.newaxis]
        Gs_gamma_batch = np.array(Gs_gamma).reshape(-1,1)  #[:, np.newaxis]

        ydl5 = np.array(actions_batch).flatten('C')
        actions_batch2 = ydl5

        ### verify. unknow network type. verify has to detect the type and shape
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud, 54)
        #_ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_12, state3s_batch, state3s_batch2, Gs_batch, Gs, Gs_gamma_batch, Gs_gamma, actions_batch2, actions_batch)
        
        #fit() for policy, action=100% and else=0. diff to fit q(s,t) which keep 'else' as it is
        #else=0, 才满足梯度公式，只有一项
        discard_possibility0 = np.eye(54)[actions_batch2]
        discard_possibility = discard_possibility0 * Gs_gamma_batch
        
        #print("pre_learn_G: ", Gs.shape, state2s_batch.shape, discard_possibility.shape) #, self.net0_list)
        if True == piv_net:
            self.sync_acquire()
            self.sync_mp_net0_to_local(self.net0_list, self.primay_net)
            his = self.primay_net.fit(state3s_batch, [discard_possibility,Gs_batch], verbose=0, batch_size=256) #11*64)
            self.sync_local_to_mp_net0(self.primay_net, self.net0_list)
            self.sync_release()

            ydl_cnt_pi += 1
            if ydl_cnt_pi % 100 == 0 :
                print("piv_net    X, Y shape ", ydl_cnt_pi, state3s_batch.shape, discard_possibility.shape, 256)
        else:
            self.sync_acquire()
            self.sync_mp_net0_to_local(self.net0_list, self.primay_net)
            his = self.primay_net.fit(state3s_batch, discard_possibility, verbose=0, batch_size=256) #11*64)
            self.sync_local_to_mp_net0(self.primay_net, self.net0_list)
            self.sync_release()
            #用q训练v, 此时应是对的，因为p(a)=100%. 如果p(a)<100%, 应概率累加所有a对应的G值=G期望值
            self.secondary_net.fit(state3s_batch, Gs_batch, verbose=0, batch_size=256) #11*64)

            ydl_cnt_pi += 1
            if ydl_cnt_pi % 100 == 0 :
                print("primay_net       X, Y shape ", ydl_cnt_pi, state3s_batch.shape, discard_possibility.shape, 256)
                print("secondary_net    X, Y shape ", ydl_cnt_pi, state3s_batch.shape, Gs_batch.shape, 256)

        return his

