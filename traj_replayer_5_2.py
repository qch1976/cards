#v5.2:   guess support 

import numpy as np

import game_verify_online_5_2 as game_verify

#############################################################
# ToDO:
# 1. priority
# 2. refactory: the storage decoupling to state/bs/action/discard, etc. just data1, data2, data3 w/o 'type'

class TrajectoryReplayer:
    def __init__(self, capacity, state_shape, next_state_shape):
        #the mempory size of state3s is huge, if multiple CPUs, such as 10 CPUs parallel.
        #one game size = (1, 11, 4, 1, 5, 54)*2bytes(float16)*(batch_size=1)=23.76k
        #200k games(regardless of batch)=4.752G, *10CPU = 47.52G !!!
        #TBD: reference to: memory realtime data compress: https://sparse.pydata.org/en/latest/quickstart.html
        #TBD: spares packet install failure!!!
        self.np_states  = np.array([], dtype=np.float16)
        self.np_actions = np.array([], dtype=np.int8)
        self.np_rewards = np.array([], dtype=np.float16)
        self.np_bs      = np.array([], dtype=np.float16)
        self.np_next_states = np.array([], dtype=np.float16)  #reused as 'fraud' in guess
        self.np_discards = np.array([], dtype=np.int8)        #for guess
        self.np_priorities  = np.array([], dtype=np.int8)
        self.state_shape = state_shape  #use (5, 54) rather than (5*54) for readable data
        self.next_state_shape = next_state_shape
        
        self.upper_mem_limitation = capacity
        return
    
    def store(self, np_states, np_actions, np_rewards, np_bs, np_next_states, np_discards=0):
        #np_state3s and np_next_states shape=(11-player*batch-info(5-54), or(4-54*2))
        #np_actions, np_rewards shape=(11-player*batch-info(0))
        #bs shape=(11-player*batch-info(0))
        np_actions = np_actions.reshape(-1)
        np_rewards = np_rewards.reshape(-1)
        np_bs = np_bs.reshape(-1)
        
        if isinstance(np_discards, int): #non guess mode
            batch_size = np_actions.shape[0]  #dummy 0
            np_discards = np.zeros((batch_size,))
        else:
            #np_discards in guess, shape=(batch, 54), same to others
            pass
            
        #state_shape = np_states.shape
        #np_states = np_states.reshape(-1, state_shape[2], state_shape[3])
        #np_next_states = np_next_states.reshape(-1, state_shape[2], state_shape[3])
        new_state_shape = (-1,) + self.state_shape
        new_next_state_shape = (-1,) + self.next_state_shape
        
        np_states = np_states.reshape(new_state_shape) #-1, state_shape[2], state_shape[3])
        np_next_states = np_next_states.reshape(new_next_state_shape) #-1, state_shape[2], state_shape[3])
        
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
            obsolete_len = int(self.np_states.shape[0] * 0.3)
            self.np_states  = np.delete(self.np_states, np.s_[0:obsolete_len], axis=0)
            self.np_actions = np.delete(self.np_actions, np.s_[0:obsolete_len], axis=0)
            self.np_rewards = np.delete(self.np_rewards, np.s_[0:obsolete_len], axis=0)
            self.np_bs      = np.delete(self.np_bs, np.s_[0:obsolete_len], axis=0)
            self.np_next_states = np.delete(self.np_next_states, np.s_[0:obsolete_len], axis=0)
            self.np_discards = np.delete(self.np_discards, np.s_[0:obsolete_len], axis=0)
            self.np_priorities = np.delete(self.np_priorities, np.s_[0:obsolete_len], axis=0)
            
        #self.xxx shape: game-round-player*bacth-info
        #np_actions = np_actions[np.newaxis,:]
        #np_rewards = np_rewards[np.newaxis,:]
        if self.np_states.shape[0] == 0:
            self.np_states  = np_states.astype(np.float16)
            self.np_actions = np_actions.astype(np.int8)
            self.np_rewards = np_rewards.astype(np.float16)
            self.np_bs      = np_bs.astype(np.float16)
            self.np_next_states = np_next_states.astype(np.float16)
            self.np_discards  = np_discards.astype(np.int8)
            self.np_priorities  = np_priorities.astype(np.int8)
        else:
            self.np_states  = np.concatenate((self.np_states, np_states.astype(np.float16)), axis=0)
            self.np_actions = np.concatenate((self.np_actions, np_actions.astype(np.int8)), axis=0)
            self.np_rewards = np.concatenate((self.np_rewards, np_rewards.astype(np.float16)), axis=0)
            self.np_bs      = np.concatenate((self.np_bs, np_bs.astype(np.float16)), axis=0)
            self.np_next_states = np.concatenate((self.np_next_states, np_next_states.astype(np.float16)), axis=0)
            self.np_discards  = np.concatenate((self.np_discards, np_discards.astype(np.int8)), axis=0)
            self.np_priorities  = np.concatenate((self.np_priorities, np_priorities.astype(np.int8)), axis=0)
    
        return self.np_states.shape[0]
    
    
    def sample(self, volume):
        length = self.np_states.shape[0]
        #sample volume from storage
        candidates = min(volume, length)
        #TBD: priority
        selected_trajectory = np.random.choice(length, size=candidates, replace=False)
        return self.np_states[selected_trajectory], self.np_actions[selected_trajectory], self.np_rewards[selected_trajectory], self.np_bs[selected_trajectory], self.np_next_states[selected_trajectory], self.np_discards[selected_trajectory]
    
    def prioritize(self, np_state3s, np_rewards): #TBD: worng prioritization here. the replayer would be used by diff agent. those will make the priority quite different. should decide the priority from agent level rather than replayer level
        #what is the priority expeirence?
        np_pri_temp = np.full(np_rewards.shape, 1, dtype=np.int8)
        return np_pri_temp

    def sample_scope(self, sample_from, sample_to):
        if -1 == sample_to:
            states, actions, rewards, bs, next_states, discards = self.np_states[sample_from:], self.np_actions[sample_from:], self.np_rewards[sample_from:], self.np_bs[sample_from:], self.np_next_states[sample_from:], self.np_discards[sample_from:]
        else:
            states, actions, rewards, bs, next_states, discards = self.np_states[sample_from:sample_to], self.np_actions[sample_from:sample_to], self.np_rewards[sample_from:sample_to], self.np_bs[sample_from:sample_to], self.np_next_states[sample_from:sample_to], self.np_discards[sample_from:sample_to]
        return states, actions, rewards, bs, next_states, discards

    def remove_scope(self, sample_from, sample_to):
        if -1 == sample_to:
            self.np_states  = np.delete(self.np_states, np.s_[sample_from:], axis=0)
            self.np_actions = np.delete(self.np_actions, np.s_[sample_from:], axis=0)
            self.np_rewards = np.delete(self.np_rewards, np.s_[sample_from:], axis=0)
            self.np_bs      = np.delete(self.np_bs, np.s_[sample_from:], axis=0)
            self.np_next_states = np.delete(self.np_next_states, np.s_[sample_from:],  axis=0)
            self.np_discards = np.delete(self.np_discards, np.s_[sample_from:], axis=0)
            self.np_priorities = np.delete(self.np_priorities, np.s_[sample_from:], axis=0)
        else:
            self.np_states  = np.delete(self.np_states, np.s_[sample_from:sample_to], axis=0)
            self.np_actions = np.delete(self.np_actions, np.s_[sample_from:sample_to], axis=0)
            self.np_rewards = np.delete(self.np_rewards, np.s_[sample_from:sample_to], axis=0)
            self.np_bs      = np.delete(self.np_bs, np.s_[sample_from:sample_to], axis=0)
            self.np_next_states = np.delete(self.np_next_states, np.s_[sample_from:sample_to], axis=0)
            self.np_discards = np.delete(self.np_discards, np.s_[sample_from:sample_to], axis=0)
            self.np_priorities = np.delete(self.np_priorities, np.s_[sample_from:sample_to], axis=0)

    def get_total_records(self):
        return self.np_states.shape[0]
    
    
def test_scope(sample_from, sample_to):
    ydl = np.arange(100)
    ydl1 = ydl[sample_from:sample_to]
    
#test_scope(10, 40) #0-39
#test_scope(0,-1) #0-98

    