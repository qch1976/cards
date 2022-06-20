import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras
import copy

#import play_agent_DNN_base_5_2 as DNN_base
import game_verify_online_5_2 as game_verify
import traj_replayer_5_2 as replayer


################################################
# TODO:
# 1. guess() alg optimize
################################################

def softmax(guess_inhand2):
    #expected shape=(n, 3, 54). n=batch, 1
    #if all guess_inhand2=-inf, a warning is throwed out. 
    #pls upper func guarantee the input guess_inhand2 not all -inf
    '''
    #import warnings
    #warnings.filterwarnings('error')
    try:
        x = guess_inhand2 - np.max(guess_inhand2, axis=1)[:,np.newaxis,:]  #+,-,*,/ : shape中只能有一个dim不等，才能操作
    except RuntimeWarning as e:
        print("YDL in softmax: ", e)

    all_inf = np.where(guess_inhand2==float('-inf'), 1, 0)
    if all_inf.all():
        #print("YDL in softmax: ")
        softmax_x = np.full((guess_inhand2.shape), np.nan)
    '''

    x = guess_inhand2 - np.max(guess_inhand2, axis=1)[:,np.newaxis,:]  #+,-,*,/ : shape中只能有一个dim不等，才能操作
    exp_x = np.exp(x)
    exp_x_sum = np.sum(exp_x, axis=1)[:,np.newaxis,:]
    softmax_x = exp_x / exp_x_sum
    return softmax_x

ydl_cnt = 0
class PlayGuess_base(): #DNN_base.PlayAgentDNNBase):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, filename_g='', learning_rate=0.00001, learning_amount=2000):
        #super().__init__(learning_rate, epsilon=0.0, gamma=0.0, net0_list=net0_list, name=[])
        
        self.learning_limit = learning_amount  #
        #self.fraud = 5 #input is state3
        self.filename_g = filename_g
        self.learning_rate = learning_rate
        return

    def integrity(self, state3s_batch):
        #shape=(n,5,54)
        
        #如果庄家state3,discard算作'known'
        state3s_batch1 = np.where(state3s_batch == 0, 0, 1)  #move fmt0/1/2/3 to (0,1)
        known_cards = np.sum(state3s_batch1, axis=1)
        unknown_cards = np.logical_not(known_cards)[:,np.newaxis,:]
        ydl1 = np.sum(known_cards, axis=1)
        ydl2 = np.sum(unknown_cards, axis=2)
        #verify
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_17, known_cards)
        return unknown_cards
    
    def players_inhand_length(self, state3s_batch, guess_env_callback):
        #shape=(n,5,54), in guess(), no -1 in played area. -1 is in the [0] onyl. only learning() has -1 in played area
        #discard is not counted in the length
        fmts = guess_env_callback(3)
        
        #3 players inhand
        played_cards = np.where(state3s_batch[:,2:] > fmts[0], 1, 0)   #discard=fmt[1] should be < 0
        not_played_cards_length = 12 - np.sum(played_cards, axis=2)
        
        return not_played_cards_length   #(100,3), 3 players inhand length include myself at [0], 沿下家出牌顺序

    # ignore the complex method. keep but NOT MAINTAINed
    # 1. was based on sigmiod() output. 事实上，不应该用sigmiod，纯Q更合适，因为有softmax
    # 2. round1 and round2 makes the length checking difficulty. the length of a player after round1 would be smaller than 0. so does round2    
    def guess_calc_0_not_done_yet_length_check_in_round12(self, state3s_batch, guess_inhand1, guess_discard1, unknown_cards, players_inhand_length, guess_env_callback): 
        # unknown_cards includes 'discard' if current is not banker
        # players_inhand_length doesn't include 'discard' regardless current is banker or not
        
        batch_size = state3s_batch.shape[0]

        #full guess results
        guess_cards = np.full((batch_size, 3, 54), 0, dtype=int)  #(0, 1)

        ######## process 'discard' ###########
        #finalize the discard card. regardless of banker or not, bankers_yes will correct it later
        #guess_discard = (guess_discard1 * unknown_cards).reshape(batch_size, 54)
        
        #replace above line
        guess_discard = np.where(unknown_cards==0, float("-inf"), guess_discard1).reshape(batch_size, 54)
        guess_discard_oindex = np.argsort(-guess_discard, axis=1)[:,:6]

        #庄家，unknown_cards里没有discard. 可以忽略guess_discard
        #非庄家，unknown_cards里有discard. 要先在unknown_cards里剪掉guess_discard
        bankers_yes = guess_env_callback(1, state3s_batch)  #current is banker. banker = True
        unknown_cards_minus_discard0 = np.zeros((batch_size, 54))
        batch_size_rep6 = np.arange(batch_size)[:,np.newaxis].repeat(6, axis=1)
        unknown_cards_minus_discard0[batch_size_rep6.reshape(-1), guess_discard_oindex.reshape(-1)] = 1   #guessed discard
        unknown_cards_minus_discard1 = unknown_cards_minus_discard0 * np.logical_not(bankers_yes)[:,np.newaxis]  #if banker=True, doesn't remove the discard
        unknown_cards_minus_discard = unknown_cards - unknown_cards_minus_discard1[:,np.newaxis,:]  #remove the discard if not a banker
        guess_inhand2 = guess_inhand1 * unknown_cards_minus_discard  #guess_inhand2 exclude discard for non-banker. but if it is banker, nothing removed since unknown_cards has not 'discard' already

        banker_pos = guess_env_callback(2) #banker_pos shape=(<batch,1>,),  =[0,1,2,3]
        #banker_pos == 0, the current player is banker. then need not to set discard card in [3,54]
        bankers_no = np.logical_not(bankers_yes)  #current is not banker. banker_no=true
        guess_cards[bankers_no, banker_pos[bankers_no]-1, :] = unknown_cards_minus_discard1[bankers_no]

        ######### ALG: make player length ##########
        # select guess card from (3,54). every player's lenght must algin to players_inhand_length
        #round1: if 1st max >= 1.618 2nd max ==> must choice the 1st max
        #round2: if 3rd max <= 10% ==> must ignore the 3rd max
        #round3: else random.choice() with reducing the player_length counter
        
        ######### round 1 #############
        # guess_inhand2 是net出来的sigmoid, (n,3,54), 有0(被unknown_cards_minus_discard相乘)
        guess_inhand2_argsorted = (np.argsort(-guess_inhand2, axis=1) * unknown_cards_minus_discard).astype(int)
        
        batch_size_rep54 = np.arange(batch_size)[:,np.newaxis].repeat(54, axis=1)
        fifty4_rep_batch = np.arange(54)[np.newaxis,:].repeat(batch_size, axis=0)
        
        max_first_arg  = guess_inhand2_argsorted[batch_size_rep54.reshape(-1), 0, fifty4_rep_batch.reshape(-1)]
        max_second_arg = guess_inhand2_argsorted[batch_size_rep54.reshape(-1), 1, fifty4_rep_batch.reshape(-1)]
        
        #select_round_1指明满足 >1.618的位置
        select_round_1 = guess_inhand2[batch_size_rep54.reshape(-1), max_first_arg, fifty4_rep_batch.reshape(-1)] > (guess_inhand2[batch_size_rep54.reshape(-1), max_second_arg, fifty4_rep_batch.reshape(-1)] * 1.618)
        select_round_1 = select_round_1.reshape(batch_size, 54)
        
        #guess_inhand存储最终的inhand. 先在所有max的位置上置1，再用select_round_1清理
        guess_inhand = np.full((batch_size, 3, 54), 0, dtype=int)  #(0, 1)
        guess_inhand[batch_size_rep54.reshape(-1), guess_inhand2_argsorted[:,0,:].reshape(-1), fifty4_rep_batch.reshape(-1)] = 1
        guess_inhand = guess_inhand * select_round_1[:,np.newaxis,:]  #>1,618, then 确定 guess result

        #unknown_cards_minus_discard减掉已选中的位置
        #guess_inhand_sum = np.sum(guess_inhand, axis=1, keepdims=True)
        unknown_cards_minus_discard -= select_round_1[:,np.newaxis,:]  #guess_inhand_sum
        guess_inhand2 *= unknown_cards_minus_discard
        
        #players_inhand_length update. 计算[0,1,2]中，各有几次排第一并且>1.618. 
        position_cnt = np.zeros((batch_size, 3), dtype=int)
        pos_0_cnt = np.where(guess_inhand2_argsorted[:,0,:]==0, 1, 0) * select_round_1
        pos_1_cnt = np.where(guess_inhand2_argsorted[:,0,:]==1, 1, 0) * select_round_1
        pos_2_cnt = np.where(guess_inhand2_argsorted[:,0,:]==2, 1, 0) * select_round_1
        position_cnt[:,0] = np.sum(pos_0_cnt, axis=1)
        position_cnt[:,1] = np.sum(pos_1_cnt, axis=1)
        position_cnt[:,2] = np.sum(pos_2_cnt, axis=1)
        players_inhand_length -= position_cnt

        ######### round 2 #############
        select_round_2 = np.logical_and((guess_inhand2 <= 0.1), (guess_inhand2 > 0.0)) #if ==0, don't be replaced by -inf
        guess_inhand2 *= np.logical_not(select_round_2)
        guess_inhand2[select_round_2] = float('-inf')  #for exp(-inf)=0

        ######### round 3 #############
        #guess_inhand2里是net出来的sigmiod. 但含有0:不存在的card（被unknow处理掉，被>1.618处理掉的）； -inf:被 <10%处理掉的
        softmax_x = softmax(guess_inhand2)

        #guess_inhand3是softmax的结果. 正常值: >0; 0 <= -inf; nan <= -inf*3
        #shape: (batch,3,54) = (batch, 3, 54) * (batch, 1, 54)
        guess_inhand3 = softmax_x * unknown_cards_minus_discard
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_19, unknown_cards_minus_discard, guess_inhand2, guess_inhand3)

        #guess card from softmax
        for i in range(batch_size): #loop=n*54. bad!!. reasonable length for all players. discarded cards?
            j = 0
            while j < 54:  #the 0~53 would be random
                if True == unknown_cards_minus_discard[i,0,j]:
                    possibles0 = guess_inhand3[i].T[j]
                    avail_possibles = np.argwhere(possibles0 > 0.0).reshape(-1)
                    possibles = possibles0[possibles0 > 0.0]
                    selected = np.random.choice(avail_possibles, p=possibles)
                    guess_inhand[i, selected, j] = 1
                    players_inhand_length[i,selected] -= 1
                    
                    if 0 == players_inhand_length[i,selected]:
                        guess_inhand2[i, selected] = float('-inf')  #possible: that all are -inf. in softmax() would warning. the result=nan*3
                        softmax_x = softmax(guess_inhand2[i][np.newaxis,:,:])
                        guess_inhand3[i] = softmax_x * unknown_cards_minus_discard[i]
                j += 1

        ######### translate fmt[0, 1, 2, 3] #############
        fmts = guess_env_callback(3)

        ydl1 = np.sum(guess_cards, axis=2)
        guess_discard_fmt = np.where(guess_cards==1, fmts[1], guess_cards)  # discard exist: 1 to fmt1

        ydl2 = np.sum(guess_inhand, axis=2)
        guess_inhand_fmt = guess_env_callback(0, guess_inhand)
        guess_cards = guess_discard_fmt.astype(float) + guess_inhand_fmt #combine 'discard' with normal [3,54]. translate from 0/1 to fmt2/3
        
        ydl3 = np.where(guess_cards!=0, 1, 0)
        ydl4 = np.sum(ydl3, axis=2)

        #verify discard position is right
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_24, guess_cards, guess_discard_oindex, state3s_batch, guess_inhand1, guess_discard1, unknown_cards, guess_env_callback)
        #verify discard position is right
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_21, guess_cards, guess_discard_oindex, state3s_batch, guess_inhand1, guess_discard1, unknown_cards, guess_env_callback)
        return guess_cards
        

    # a simple method replaced
    # 1. based on Q. net changed. 纯Q更合适，因为有softmax
    # 2. ignore round1 and round2, just take the possibility of softmax(Q)
    def guess_calc(self, state3s_batch, guess_inhand1, guess_discard1, unknown_cards, players_inhand_length, guess_env_callback): 
        # unknown_cards includes 'discard' if current is not banker
        # players_inhand_length doesn't include 'discard' regardless current is banker or not
        
        batch_size = state3s_batch.shape[0]

        #full guess results
        guess_cards = np.full((batch_size, 3, 54), 0, dtype=int)  #(0, 1)

        ######## process 'discard' ###########
        # finalize the discard card, max(6)
        # input: 
        #   guess_discard1： Q, not process by unknown yet
        #   unknown_cards
        #   guess_inhand1: Q, guessed inhand cards w/o discard. not process by unknown yet
        # output:
        #   guess_cards: indicate discard card (0,1)
        #   unknown_cards_minus_discard: updated 'unknown_cards'
        #   guess_inhand2: Q, guessed inhand cards w/o discard, process by unknown already
        #
        # !!NOTE: 如果activation=None， Q可能为负值，在argsort()时，0可能被排在前6名，但实际上是不该有的card
        #         fit()的y值为 <0,1>, 所以用sigmiod（）做activation,并不会大大改变output的值域.
        #      ==> DISC1: [unknown] = -inf
        #      ==> DISC2: sigmiod replace none as activation. PR#56: sigmiod(）的输出，可能有0值， 必须用-inf再替代一次
        
        #unknown_cards_inf = np.where(unknown_cards==False, float('-inf'), 0)       #reserve these 2 lines, recover when DISC1
        #guess_discard = (guess_discard1 + unknown_cards_inf).reshape(batch_size, 54)
        
        #guess_discard = (guess_discard1 * unknown_cards).reshape(batch_size, 54)    #reserve this line.  recover when DISC2 
        #replace above line(DISC2)
        guess_discard = np.where(unknown_cards==0, float("-inf"), guess_discard1).reshape(batch_size, 54)
        guess_discard_oindex = np.argsort(-guess_discard, axis=1)[:,:6] #regardless of banker or not, bankers_yes will correct it later

        #庄家，unknown_cards里没有discard. 可以忽略guess_discard
        #非庄家，unknown_cards里有discard. 要先在unknown_cards里剪掉guess_discard
        bankers_yes = guess_env_callback(1, state3s_batch)  #current is banker. banker = True
        unknown_cards_minus_discard0 = np.zeros((batch_size, 54), dtype=int)
        batch_size_rep6 = np.arange(batch_size)[:,np.newaxis].repeat(6, axis=1)
        unknown_cards_minus_discard0[batch_size_rep6.reshape(-1), guess_discard_oindex.reshape(-1)] = 1   #guessed discard
        unknown_cards_minus_discard1 = unknown_cards_minus_discard0 * np.logical_not(bankers_yes)[:,np.newaxis]  #if banker=True, doesn't remove the discard
        unknown_cards_minus_discard = unknown_cards - unknown_cards_minus_discard1[:,np.newaxis,:]  #remove the discard if not a banker
        guess_inhand2 = guess_inhand1 * unknown_cards_minus_discard  #guess_inhand2 exclude discard for non-banker. but if it is banker, nothing removed since unknown_cards has not 'discard' already
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_26, unknown_cards_minus_discard, bankers_yes, unknown_cards_minus_discard0, unknown_cards_minus_discard1, state3s_batch, unknown_cards, guess_discard1, guess_discard_oindex)

        banker_pos = guess_env_callback(2) #banker_pos shape=(<batch,1>,),  =[0,1,2,3]
        #banker_pos == 0, the current player is banker. then need not to set discard card in [3,54]
        bankers_no = np.logical_not(bankers_yes)  #current is not banker. banker_no=true
        guess_cards[bankers_no, banker_pos[bankers_no]-1, :] = unknown_cards_minus_discard1[bankers_no]

        ######### ALG: choice card with limitation of player length ##########
        # input
        #   guess_inhand2: Q, 但含有0:不存在的card（被unknow处理掉的)
        #
        # output:
        #   guess_inhand2: Q, choice()结束的player变为-inf， 目的是softmax(-inf)=0
        #   guess_inhand: selected cards
        #
        # loop oindex one by one. if a player's card all allocated, set -inf to inhand2[] and re-softmax to inhand3[]
        
        ### softmax processing. softmax(0)的结果，是平均possibility
        softmax_x = softmax(guess_inhand2)

        #shape: (batch,3,54) = (batch, 3, 54) * (batch, 1, 54)
        guess_inhand3 = softmax_x * unknown_cards_minus_discard
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_19, unknown_cards_minus_discard, guess_inhand2, guess_inhand3)

        #guess_inhand存储最终的inhand
        guess_inhand = np.full((batch_size, 3, 54), 0, dtype=int)  #(0, 1)

        ### with the limitation of length, choice() based off softmax() possibiliy
        for i in range(batch_size): #loop=n*54. bad!!. reasonable length for all players. discarded cards?
            j = 0
            while j < 54:  #the 0~53 could be random
                if True == unknown_cards_minus_discard[i,0,j]:
                    possibles0 = guess_inhand3[i].T[j]
                    avail_possibles = np.argwhere(possibles0 > 0.0).reshape(-1)
                    possibles = possibles0[possibles0 > 0.0]
                    selected = np.random.choice(avail_possibles, p=possibles)
                    guess_inhand[i, selected, j] = 1
                    players_inhand_length[i,selected] -= 1
                    
                    if 0 == players_inhand_length[i,selected]:
                        guess_inhand2[i, selected] = float('-inf')  #possible: that all are -inf. in softmax() would warning. the result=nan*3
                        if np.sum(players_inhand_length[i]) > 0:
                            softmax_x = softmax(guess_inhand2[i][np.newaxis,:,:])
                            guess_inhand3[i] = softmax_x * unknown_cards_minus_discard[i]
                        else:
                            break  #break from j loop
                j += 1

        ######### translate fmt[0, 1, 2, 3] #############
        fmts = guess_env_callback(3)

        ydl1 = np.sum(guess_cards, axis=2)
        guess_discard_fmt = np.where(guess_cards==1, fmts[1], guess_cards)  # discard exist: 1 to fmt1. dont via guess_env_callback(0) since trump distribution is known but discard is guessed

        ydl2 = np.sum(guess_inhand, axis=2)
        guess_inhand_fmt = guess_env_callback(0, guess_inhand)
        guess_cards = guess_discard_fmt.astype(float) + guess_inhand_fmt #combine 'discard' with normal [3,54]. translate from 0/1 to fmt2/3
        
        ydl3 = np.where(guess_cards!=0, 1, 0)
        ydl4 = np.sum(ydl3, axis=2)

        #verify discard position is right
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_24, guess_cards, guess_discard_oindex, state3s_batch, guess_inhand1, guess_discard1, unknown_cards, guess_env_callback)
        return guess_cards

    def guess(self, state3s_batch, unknown_cards, players_inhand_length, guess_env_callback):
        #state3s_batch shape=(batch-info(5*54)) or info(5,54,1). upper func make sure the shape
        batch_size = state3s_batch.shape[0]

        #input=(0, -1, 0.5, 1)
        #output = (0,1) rather than fmt0/1/2/3. reshape in callback func() who knows envs
        guess_inhand0, guess_discard0 = self.guess_net.predict(state3s_batch)  #out shape(batch-3-54). x[;0]=(-1, 0, 0.5, 1)="net fmt"
        guess_inhand1 = guess_inhand0.reshape(batch_size, 3, 54) #与in的对应顺正确， 还没有体现 player*batch的转换
        guess_discard1 = guess_discard0[:,np.newaxis,:]
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_18, guess_inhand0, guess_inhand1)

        players_inhand_length_ut = copy.deepcopy(players_inhand_length)
        guess_cards = self.guess_calc(state3s_batch, guess_inhand1, guess_discard1, unknown_cards, players_inhand_length, guess_env_callback)
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_25, guess_cards, state3s_batch, guess_inhand1, guess_discard1, unknown_cards, players_inhand_length_ut, guess_env_callback)
        return guess_cards

    def guess_learning_single_round(self, state3s_batch, state3s_fraud_batch, discard_batch):
        #upper func need to reshape the state3s_batch. DNN or CNN
        length = state3s_batch.shape[0]
        dummy = np.zeros((length))
        history = []
        
        #here, state3s_batch and state3s_fraud_batch are different in shape.
        #1404bytes=1(5*54)+1(4*54*2). 200*1404 = 280kbytes
        stored_length = self.replayer.store(state3s_batch, dummy, dummy, dummy, state3s_fraud_batch, discard_batch)
        
        if self.learning_limit <= stored_length:
            history = self.flush_learning()
        return history

    def flush_learning(self):
        global ydl_cnt
        learning_state3s, _, _, _, learning_state3s_fraud, learning_discard = self.replayer.sample_scope(0, -1)
        batch_size = learning_state3s.shape[0]
        history = []
        if batch_size > 0:   #in last flush(), if 2 more agents have same 'name' and net, batch_size of non-first agent would be 0.
            learning_state3s1 = self.shapes(learning_state3s)  #defined ub upper()
            learning_state3s_fraud1 = np.where(learning_state3s_fraud==0, 0, 1)  #fit(): x=(-1, 0, 0.5, 1)="net fmt"; y=(0,1), present the possibility
            #learning_state3s1 shape(n-54*5)-DNN or (n-5-54-1)-CNN
            #learning_state3s_fraud shape (n-4*54*2), whatever CNN or DNN
            history = self.guess_net.fit(learning_state3s1, [learning_state3s_fraud1[:, 1:, 0:54].reshape(batch_size, -1), learning_discard], verbose=0, batch_size=256)

            ydl_cnt += 1
            if ydl_cnt % 1000 == 0 :
                print("guess_net    X, Y shape ", ydl_cnt, learning_state3s1.shape, learning_state3s_fraud[:, 1:, 0:54].reshape(batch_size, -1).shape, 256)
    
            self.replayer.remove_scope(0, -1)
        
        return history
        
    def guess_learning_multi_games(self):
        return
    
    def guess_learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #input state3: shape: round(=11)-player*batch-info(5, 54), or (4, 54*2)
        #action, reward shape: round(=11)-player*batch-info(0). no change in this method
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)
        return

    def save_model(self):
        self.guess_net.save(self.filename_g)
        print("guess model saved: ", self.filename_g)

    def load_model(self):
        net = keras.models.load_model(self.filename_g) #, custom_objects={'my_loss': my_loss}) #{'ydl_loss': ydl_loss}, NAME MUST BE SAME
        print("guess model loaded: ", self.filename_g)
        return net

    def new_state3_translate(self, state3s_batch, guessed_state3s):
        #new_state3s_batch shape=(batch-info(4-108)). translate to fraud shape
        batch = state3s_batch.shape[0]
        new_state3s_batch = np.zeros((batch, 4, 54*2))
        new_state3s_batch[:, 0,  0:54]    = state3s_batch[:, 0]
        new_state3s_batch[:, :,  54:54*2] = state3s_batch[:, 1:]
        new_state3s_batch[:, 1:, 0:54]    = guessed_state3s
        return new_state3s_batch
