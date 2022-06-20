import time
import numpy as np
import tensorflow.compat.v2 as tf2


import play_agent_DNN_base_5_2 as DNN_base
import DNN_network_utils_5_2 as DNN_net


class PlayAgentDNN_MC_pi(DNN_base.PlayAgentDNNBase):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, 
                 reload=False, net0_list=0, name=['non-behavior'], loss1=tf2.losses.categorical_crossentropy, fraud=5):
        super().__init__(learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name)
        
        self.filename_pi = filename_e
        self.filename_v  = filename_t
        self.fraud = fraud
        if ( reload == True ):
            self.policy_net = self.load_model(self.filename_pi, loss1)
            self.primay_net = self.policy_net
            self.v_net = self.load_model(self.filename_v)
            self.secondary_net = self.v_net
            self.policy_net.summary()
            self.v_net.summary()
        else:
            '''
            if 4 == fraud:
                input_size=4*54*2
            else:
                input_size=5*54
            input_shape=(input_size,)
            '''
            ###############
            # net=pi(s)
            ###############
            input_size, input_shape = DNN_net.reshape_DNN_network(self.fraud)
            output_size=54
            activation=tf2.nn.relu
            loss=loss1 #tf.losses.categorical_crossentropy  # tf.losses.mse
            output_activation=tf2.nn.softmax
            self.policy_net = DNN_net.build_network(input_size, input_shape, hidden_layers, output_size,
                                                    activation, loss, output_activation, learning_rate)
            self.primay_net = self.policy_net

            ###############
            # net=v(s)
            ###############
            output_size=1
            activation=tf2.nn.relu
            loss=tf2.losses.mse
            output_activation=None
            self.v_net = DNN_net.build_network(input_size, input_shape, hidden_layers, output_size,
                                               activation, loss, output_activation, learning_rate)
            self.secondary_net = self.v_net

            self.first_net0_sync()
            '''
            if net0_list != 0 :
                print("PlayAgentDNN_MC_pi: net0 id1 + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])

                if (7 == net0_list[0][0][0,1] and 3 == net0_list[0][2][50,50] and 7 == net0_list[0][4][126,2]):
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("PlayAgentDNN_MC_pi: first copy to net0")
                    self.sync_acquire()
                    self.sync_local_to_mp_net0(self.policy_net, net0_list)
                    self.sync_release()
                else:
                    print("PlayAgentDNN_MC_pi: second+ copy to local")
                    self.sync_acquire()
                    self.sync_mp_net0_to_local(net0_list, self.policy_net)
                    self.sync_release()
            '''
        return

    def decide(self, state3s_batch, available_mask_batch, train=True):
        batch_size = state3s_batch.shape[0]
        return self.decide_pi(state3s_batch.reshape(batch_size,-1), available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return

    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):
        #shape: round(=11)-players*batch-info
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)
        if 4 == self.fraud:
            state3s_batch2 = state3s_batch.reshape(-1, 4*54*2)
        else:
            state3s_batch2 = state3s_batch.reshape(-1, 5*54)
        
        #verify TBD
        his = self.learning_single_game_pi(state3s_batch2, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)
        return his

    def save_models(self):
        super().save_model(self.policy_net, self.filename_pi)
        super().save_model(self.v_net, self.filename_v)


class PlayAgentDNN_MC_pi_Behavior(PlayAgentDNN_MC_pi):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior'], fraud=5):
        #self.agent_b = RandomAgent()
        
        def my_loss(y_true, y_pred): #YDL: 就是对loss=dot(),对theta求梯度 
            # y_true = y = (df['psi'] / df['behavior']) = (gamma^t * Gt) / b(A|S)
            # y_pred = pi(A|S,theta)
            # - y_true * y_pred 就是ch7.3的theta更新公式. 这样用tf的loss函数， miao!!!
            # loss(theta),对theta求梯度 = -(gamma^t * Gt) / b(A|S) * 梯度(pi(A|S,theta))
            loss_b = -tf2.reduce_sum(y_true * y_pred, axis=-1)
            return loss_b
        
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma, reload=reload, net0_list=net0_list, name=name, loss1=my_loss, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentDNN_MC_pi_Behavior, train=train)
        '''
        if True == train :
            #action_oindex, b = self.agent_b.decide(available_mask_batch)
            action_oindex, b = self.decide_b(state3s_batch, available_mask_batch)
        else:
            batch_size = state3s_batch.shape[0]
            b = [1]*batch_size #b=dummy for behavior
            action_oindex = PlayAgentDNN_MC_pi.decide(self, state3s_batch, available_mask_batch, train=False)

        return action_oindex, b
        '''
    
ydl_cnt = 0
class PlayAgentDNN_MC_pi_aac(PlayAgentDNN_MC_pi): #Advanced AC, or A2C
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'AC'], loss1=tf2.losses.categorical_crossentropy, fraud=5):  #apply gradient by apply_gradient() rather than 'fit()'. the loss() is not used at all.
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma, reload=reload, net0_list=net0_list, name=name, loss1=loss1, fraud=fraud)
        #self.policy_net = actor
        #self.v_net = critic-V(S)
        
        discount = [ gamma ** i for i in range(11)]
        self.discount = np.array(discount)
    
    def shapes(self, states_batch, next_states_batch):
        state_shape = (states_batch.shape[1], -1)
        next_state_shape = (next_states_batch.shape[1], -1)
        return state_shape, next_state_shape
        
    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0, piv_net=False):
        global ydl_cnt

        #shape: round(=11)-players*batch-info
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)

        #时序差分 ~= 优势函数。 loss = - gamma_^t *[U(S)-V(S)]Ln[pi(A|S)]. U(S)=R(S) + rV(S')
        actions_batch = actions_batch.astype(np.int32)  #tf slice need int32
        state_shape = states_batch.shape 
        state_round_shape, next_state_round_shape = self.shapes(states_batch, next_states_batch)

        for round_id in range(state_shape[0]): #loop per round
                
            ##### available oindex ########
            next_bitmasks_like = np.where(next_states_batch[round_id,:,0,:]>0, 1, 0)[:,:54]  #it is not actual bitmask since no info about (1) 'first played card' even the 'first player' is known. (2)'trump', 'round' in env. agent doesn't know any env data
            learning_next_action_n = np.sum(next_bitmasks_like, axis=1)
            dones = np.where(learning_next_action_n>1, 0 ,1)  #True mean done
    
            ##### V(S') ######## shape=(player*batch-1)
            if True == piv_net: #Residual net reuse
                _, next_Vs = self.policy_v_net.predict(next_states_batch[round_id].reshape(next_state_round_shape))
            else:
                next_Vs = self.v_net.predict(next_states_batch[round_id].reshape(next_state_round_shape))
            
            ##### U(S) ########, 半梯度，U不在tape中计算
            Us = rewards_batch[round_id] + self.gamma * next_Vs[:,0] * (1. - dones)
    
            ##### V(S) ######## shape=(player*batch-1)
            if True == piv_net: #Residual net reuse
                _, Vs = self.policy_v_net.predict(states_batch[round_id].reshape(state_round_shape))
            else:
                Vs = self.v_net.predict(states_batch[round_id].reshape(state_round_shape))
    
            ##### error ######## shape=(player*batch)
            td_error = Us - Vs[:,0]  #正负皆可能，导致loss有正有负, 非凹函数? 梯度向哪个方向?
    
            np_actions_round = np.concatenate((np.arange(state_shape[1]).reshape(-1, 1), actions_batch[round_id].reshape(-1, 1)), axis=1)
            
            # actor net
            states_batch_tensor = tf2.convert_to_tensor(states_batch[round_id].reshape(state_round_shape), dtype=tf2.float32)
            Us_tensor = tf2.convert_to_tensor(Us)
            with tf2.GradientTape() as tape:
                #pi(A|S)
                if True == piv_net: #Residual net reuse
                    pi_tensor0, v_tensor0 = self.policy_v_net(states_batch_tensor)
                    mse = tf2.keras.losses.MeanSquaredError()
                    v_tensor = v_tensor0[:,0]
                else:
                    pi_tensor0 = self.policy_net(states_batch_tensor)
                pi_tensor = tf2.gather_nd(pi_tensor0, np_actions_round)
                
                #Ln[pi(A|S)]
                logpi_tensor = tf2.math.log(tf2.clip_by_value(pi_tensor, 1e-9, 1.))

                # -gamma_^t *[U(S)-V(S)]Ln[pi(A|S)]
                loss_tensor0 = -self.discount[round_id] * td_error * logpi_tensor #it is possible as 'negtive loss'
                # avarge the batch
                loss_pi_tensor = tf2.reduce_mean(loss_tensor0)  #here, negtive and positive value mixed. will it reduce the gradient?
                
                if True == piv_net: #Residual net reuse
                    loss_v_tensor = mse(Us_tensor, v_tensor)
                

            if True == piv_net: #Residual net reuse
                # both actor and critic net
                grad_tensors = tape.gradient([loss_pi_tensor, loss_v_tensor], self.policy_v_net.trainable_variables) #variables=trainable_variables in keras model()
                ydl_op = self.policy_v_net.optimizer.apply_gradients(zip(grad_tensors, self.policy_v_net.trainable_variables)) # 更新执行者网络. optimizer() would minimize the loss to negtive value
                history = []
            else:
                #梯度朝loss的最小值移动，无论这个最小值是正or负. refer to tape_test3()
                grad_tensors = tape.gradient(loss_pi_tensor, self.policy_net.trainable_variables) #variables=trainable_variables in keras model()
                ydl_op = self.policy_net.optimizer.apply_gradients(zip(grad_tensors, self.policy_net.trainable_variables)) # 更新执行者网络. optimizer() would minimize the loss to negtive value
        
                # critic net
                history = self.v_net.fit(states_batch[round_id].reshape(state_round_shape), Us, verbose=0, batch_size=256) # 更新评论者网络

                ydl_cnt += 1
                if ydl_cnt % 100 == 0 :
                    print("acc v_net    X, Y shape ", ydl_cnt, states_batch[round_id].reshape(state_round_shape).shape, Us.shape, 256)

            def tape_test3(): #loss=neg
                TRAIN_STEPS=200
                
                # Prepare train data
                train_X = np.linspace(0, 2*3.14, 1000)
                train_Y = train_X + np.random.randn(*train_X.shape)
                
                print(train_X.shape)
                
                ww=tf2.Variable(initial_value=1.0)
                bb=tf2.Variable(initial_value=1.0)
                
                optimizer=tf2.keras.optimizers.SGD(0.1)
                train_Y_tensor = tf2.convert_to_tensor(train_Y)
                for i in range(TRAIN_STEPS):
                    print("epoch:",i)
                    #print("w:", w.numpy())
                    #print("b:", b.numpy())
                    #计算和更新梯度
                    with tf2.GradientTape() as tape:
                        fn = ww * train_Y + bb
                        logit_tensor = tf2.clip_by_value(fn, 1e-9, 2*3.14)
                        loss0=tf2.math.cos(logit_tensor)
                        loss = tf2.reduce_mean(loss0)
                    gradients=tape.gradient(target=loss,sources=[ww,bb])  #计算梯度
                    #print("gradients:",gradients)
                    #print("zip:\n",list(zip(gradients,[w,b])))
                    optimizer.apply_gradients(zip(gradients,[ww,bb]))     #更新梯度
                    print("0-w ", gradients[0].numpy(), ww.numpy())
                    print("1-b", gradients[1].numpy(), bb.numpy())
                    print("loss: ", loss)  #the loss is close to -1.0, the min of cos(). so optimizer() will move forward to the min value regardless of pos or neg
            

        return history


class PlayAgentDNN_MC_pi_aac_elibility(PlayAgentDNN_MC_pi_aac): #Advanced AC
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=[0.2, 0.9, 0.9], reload=False, net0_list=0, name=['non-behavior', 'AC'], fraud=5):
        #gamma contain a-c lambda. bad
        gamma0 = gamma[0]
        actor_lambda = gamma[1]
        critic_lambda = gamma[2]
        
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma0, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        #self.policy_net = actor
        #self.v_net = critic-V(S)

        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.actor_traces = [np.zeros_like(weight) for weight in self.policy_net.get_weights()]
        self.critic_traces = [np.zeros_like(weight) for weight in self.v_net.get_weights()]

    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):
        #shape: round(=11)-players*batch-info
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)
        history = []
        
        #时序差分 ~= 优势函数。 loss = - gamma_^t *[U(S)-V(S)]Ln[pi(A|S)]. U(S)=R(S) + rV(S')
        actions_batch = actions_batch.astype(np.int32)  #tf slice need int32
        state_shape = states_batch.shape
        state_round_shape, next_state_round_shape = self.shapes(states_batch, next_states_batch)

        for round_id in range(state_shape[0]): #loop per round
                
            ##### available oindex ########
            next_bitmasks_like = np.where(next_states_batch[round_id,:,0,:]>0, 1, 0)[:,:54]  #it is not actual bitmask since no info about (1) 'first played card' even the 'first player' is known. (2)'trump', 'round' in env. agent doesn't know any env data
            learning_next_action_n = np.sum(next_bitmasks_like, axis=1)
            dones = np.where(learning_next_action_n>1, 0 ,1)  #True means done
    
            ##### V(S') ######## shape=(player*batch-1)
            next_Vs = self.v_net.predict(next_states_batch[round_id].reshape(next_state_round_shape))
            
            ##### U(S) ########. doesn't gradient 'U'
            Us = rewards_batch[round_id] + self.gamma * next_Vs[:,0] * (1. - dones)
    
            ##### V(S) ######## shape=(player*batch-1)
            Vs = self.v_net.predict(states_batch[round_id].reshape(state_round_shape))
    
            ##### error ######## shape=(player*batch)
            td_error = np.mean(Us - Vs[:,0])  #正负皆可能，导致loss有正有负, 非凹函数? 梯度向哪个方向?
    
            np_actions_round = np.concatenate((np.arange(state_shape[1]).reshape(-1, 1), actions_batch[round_id].reshape(-1, 1)), axis=1)
            
            ###################
            # actor net
            ###################
            states_batch_tensor = tf2.convert_to_tensor(states_batch[round_id].reshape(state_round_shape), dtype=tf2.float32)
            with tf2.GradientTape() as tape:
                #pi(A|S)
                pi_tensor0 = self.policy_net(states_batch_tensor)
                pi_tensor = tf2.gather_nd(pi_tensor0, np_actions_round)
                
                #Ln[pi(A|S)]
                logpi_tensor = tf2.math.log(tf2.clip_by_value(pi_tensor, 1e-9, 1.))

                # avarge the batch
                loss_tensor = tf2.reduce_mean(logpi_tensor)

            #梯度(Ln[pi(A|S)]). 梯度朝loss的最小值移动，无论这个最小值是正or负. refer to tape_test3()
            grad_tensors_pi = tape.gradient(loss_tensor, self.policy_net.trainable_variables) #variables=trainable_variables in keras model()

            #执行者资格迹: z = gamma * lamda * z + gamma_^t * 梯度(Ln[pi(A|S)])
            self.actor_traces = [self.gamma * self.actor_lambda * trace +  self.discount[round_id] * grad.numpy() for trace, grad in zip(self.actor_traces, grad_tensors_pi)]
            
            #update the gradient by elibility trace. the minus sign(-1) is here!!! :  new 梯度 = - [U(S)-V(S)] * z
            actor_grads = [tf2.convert_to_tensor(-td_error * trace, dtype=tf2.float32) for trace in self.actor_traces]
            ydl_op = self.policy_net.optimizer.apply_gradients(zip(actor_grads, self.policy_net.trainable_variables)) # 更新执行者网络. optimizer() would minimize the loss to negtive value

    
            ####################
            # critic net
            ###################
            with tf2.GradientTape() as tape:
                # V(S)
                v_tensor = self.v_net(states_batch_tensor)
            
            #梯度(V(S))
            grad_tensors_v = tape.gradient(v_tensor, self.v_net.trainable_variables)
            
            #评论者资格迹: z = gamma * lamda * z + 梯度(V(S))
            self.critic_traces = [self.gamma * self.critic_lambda * trace + self.discount[round_id] * grad.numpy() for trace, grad in zip(self.critic_traces, grad_tensors_v)]
            
            #update gradient: new 梯度 = - [U(S)-V(S)] * z
            critic_grads = [tf2.convert_to_tensor(-td_error * trace, dtype=tf2.float32) for trace in self.critic_traces]
            ydl_op = self.v_net.optimizer.apply_gradients(zip(critic_grads, self.v_net.trainable_variables))


            ###################
            # rest trace for next game
            ###################
            if dones.all(): #must DONE at same time since it is MC procedure
                self.actor_traces = [np.zeros_like(weight) for weight in self.policy_net.get_weights()]
                self.critic_traces = [np.zeros_like(weight) for weight in self.v_net.get_weights()]

        return history


class PlayAgentDNN_MC_pi_aac_Behavior(PlayAgentDNN_MC_pi_aac): #Advanced AC
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'AC'], fraud=5):
        def my_loss(y_true, y_pred): #?? 不能用fit() + my_loss()。 否则U会被计算梯度??? A: U和pi的net不同，参数不同，不会对U求梯度!=>可以用
            # y_true = [gamma^t * (Ut-V(S))] / b(A|S)
            # y_pred = pi(A|S,theta)
            # loss(theta) = -[gamma^t * (Ut-V(S))] / b(A|S) * pi(A|S,theta)
            loss_b = -tf2.reduce_sum(y_true * y_pred, axis=-1)
            return loss_b

        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud) #skip my_loss()
        #self.policy_net = actor
        #self.v_net = critic-Q(S,A)


    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentDNN_MC_pi_aac_Behavior, train=train)
        '''
        if True == train :
            #action_oindex, b = self.agent_b.decide(available_mask_batch)
            action_oindex, b = self.decide_b(state3s_batch, available_mask_batch)
        else:
            batch_size = state3s_batch.shape[0]
            b = [1]*batch_size #b=dummy for behavior
            action_oindex = PlayAgentDNN_MC_pi_aac.decide(self, state3s_batch, available_mask_batch, train=False)

        return action_oindex, b
        '''

    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):
        #shape: round(=11)-players*batch-info
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)
        history = []
        
        #时序差分 ~= 优势函数。 loss = - gamma_^t *[U(S)-V(S)]/b(A|S)*pi(A|S). U(S)=R(S) + rV(S')
        actions_batch = actions_batch.astype(np.int32)  #tf slice need int32
        state_shape = states_batch.shape 
        state_round_shape, next_state_round_shape = self.shapes(states_batch, next_states_batch)

        for round_id in range(state_shape[0]): #loop per round
                
            ##### available oindex ########
            next_bitmasks_like = np.where(next_states_batch[round_id,:,0,:]>0, 1, 0)[:,:54]  #it is not actual bitmask since no info about (1) 'first played card' even the 'first player' is known. (2)'trump', 'round' in env. agent doesn't know any env data
            learning_next_action_n = np.sum(next_bitmasks_like, axis=1)
            dones = np.where(learning_next_action_n>1, 0 ,1)  #True mean done

            ##### V(S') ######## shape=(player*batch-1)
            next_Vs = self.v_net.predict(next_states_batch[round_id].reshape(next_state_round_shape))
            
            ##### U(S) ########, 半梯度，U不在tape中计算
            Us = rewards_batch[round_id] + self.gamma * next_Vs[:,0] * (1. - dones)
    
            ##### V(S) ######## shape=(player*batch-1)
            Vs = self.v_net.predict(states_batch[round_id].reshape(state_round_shape))
    
            ##### error/b ######## shape=(player*batch)
            td_error = (Us - Vs[:,0]) / behaviors_batch[round_id]  #正负皆可能，导致loss有正有负, 非凹函数? 梯度向哪个方向?
    
            np_actions_round = np.concatenate((np.arange(state_shape[1]).reshape(-1, 1), actions_batch[round_id].reshape(-1, 1)), axis=1)
            states_batch_tensor = tf2.convert_to_tensor(states_batch[round_id].reshape(state_round_shape), dtype=tf2.float32)
            ##################
            # actor net
            ##################
            states_batch_tensor = tf2.convert_to_tensor(states_batch[round_id].reshape(state_round_shape), dtype=tf2.float32)
            with tf2.GradientTape() as tape:
                #pi(A|S)
                pi_tensor0 = self.policy_net(states_batch_tensor)
                pi_tensor = tf2.gather_nd(pi_tensor0, np_actions_round)

                # -gamma_^t *[U(S)-V(S)]/b(A|S)*pi(A|S)
                loss_tensor0 = -self.discount[round_id] * td_error * pi_tensor #it is possible as 'negtive loss'
                # avarge the batch
                loss_tensor = tf2.reduce_mean(loss_tensor0)  #here, negtive and positive value mixed. will it reduce the gradient?

            #梯度朝loss的最小值移动，无论这个最小值是正or负. refer to tape_test3()
            grad_tensors = tape.gradient(loss_tensor, self.policy_net.trainable_variables) #variables=trainable_variables in keras model()
            ydl_op = self.policy_net.optimizer.apply_gradients(zip(grad_tensors, self.policy_net.trainable_variables)) # 更新执行者网络. optimizer() would minimize the loss to negtive value
    
            ##################
            # critic net
            ##################
            # rho = pi(A|S)/b(A|S)
            pi0 = self.policy_net.predict(states_batch[round_id].reshape(state_round_shape))
            pi = pi0[np.arange(states_batch.shape[1]), actions_batch[round_id]]
            rho = pi / behaviors_batch[round_id]
            
            #fit(.., Us*rho, ..) is wrong. mse(y_true, y_pred): y_true = Us_true*rho, y_pred=Us_pre. 不再是方差计算
            with tf2.GradientTape() as tape:
                #V(S)
                Vs_tensor = self.v_net(states_batch_tensor)[:,0]  #squeeze the axis=-1

                # loss = - pi(A|S)/b(A|S) * [U(S)-V(S)]^2 
                loss_tensor0 = tf2.square((Us - Vs_tensor))
                loss_tensor1 = - rho * loss_tensor0
                
                # avarge the batch
                loss_tensor = tf2.reduce_mean(loss_tensor1)  #here, negtive and positive value mixed. will it reduce the gradient?

            #梯度朝loss的最小值移动，无论这个最小值是正or负. refer to tape_test3()
            grad_tensors = tape.gradient(loss_tensor, self.v_net.trainable_variables) #variables=trainable_variables in keras model()
            ydl_op = self.v_net.optimizer.apply_gradients(zip(grad_tensors, self.v_net.trainable_variables)) # 更新执行者网络. optimizer() would minimize the loss to negtive value


        return history



#the only difference in agent is fraud=4 rather than 5
class PlayAgentDNN_MC_pi_fraud(PlayAgentDNN_MC_pi):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, 
                 reload=False, net0_list=0, name=['non-behavior', 'fraud'], loss1=tf2.losses.categorical_crossentropy, fraud=4):
        PlayAgentDNN_MC_pi.__init__(self, hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, 
                                            epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, loss1=loss1, fraud=fraud)
        #print(PlayAgentDNN_MC_pi_fraud.__bases__)

    #add explict method with identical code: easy to understand the class invoking path
    def decide(self, state3s_fraud_batch, available_mask_batch, train=True):
        return super().decide(state3s_fraud_batch, available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return super().learning_multi_games()
    
    def learning_single_game(self, state3s_fraud_batch, actions_batch, rewards_batch, behaviors_batch=0):
        #shape: round(=11)-player*batch-info
        return super().learning_single_game(state3s_fraud_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)
            

class PlayAgentDNN_MC_pi_Behavior_fraud(PlayAgentDNN_MC_pi_Behavior): #PlayAgentDNN_MC_pi_fraud): #, PlayAgentDNN_MC_pi_Behavior):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'fraud'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        #PlayAgentDNN_MC_pi_Behavior
        
    #add explict method with identical code: easy to understand the class object in debug
    def decide(self, state3s_fraud_batch, available_mask_batch, train=True):
        return super().decide(state3s_fraud_batch, available_mask_batch, train=train)
        #PlayAgentDNN_MC_pi_Behavior
        



class PlayAgentDNN_MC_pi_aac_fraud(PlayAgentDNN_MC_pi_aac):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'fraud', 'AC'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)
            
    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):
        return super().learning_single_game(states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=behaviors_batch)
    
    
    
    
class PlayAgentDNN_MC_pi_aac_elibility_fraud(PlayAgentDNN_MC_pi_aac_elibility):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=[0.2, 0.9, 0.9], reload=False, net0_list=0, name=['non-behavior', 'fraud', 'AC'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
    
    
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)

    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):
        return super().learning_single_game(states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=behaviors_batch)
    
    
    
class PlayAgentDNN_MC_pi_aac_Behavior_fraud(PlayAgentDNN_MC_pi_aac_Behavior):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'fraud', 'AC'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, 
                         gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud) #skip my_loss()

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)


    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):
        return super().learning_single_game(states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=behaviors_batch)



    
''' #test
gamma = 0.9
discount = [ gamma ** i for i in range(11)]
discounts = np.array(discount, dtype=np.float64)
'''


def tape_test1():
    #-------------------一元梯度案例---------------------------
    print("一元梯度")
    x=tf2.constant(value=3.0)
    with tf2.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
        tape.watch(x)
        y1=2*x
        y2=x*x+2
        y3=x*x+2*x
    #一阶导数
    dy1_dx=tape.gradient(target=y1,sources=x)
    dy2_dx = tape.gradient(target=y2, sources=x)
    dy3_dx = tape.gradient(target=y3, sources=x)
    print("dy1_dx:",dy1_dx)
    print("dy2_dx:", dy2_dx)
    print("dy3_dx:", dy3_dx)

    # # -------------------二元梯度案例---------------------------
    print("二元梯度")
    x = tf2.constant(value=3.0)
    y = tf2.constant(value=2.0)
    with tf2.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
        tape.watch([x,y])
        z1=x*x*y+x*y
    # 一阶导数
    dz1_dx=tape.gradient(target=z1,sources=x)
    dz1_dy = tape.gradient(target=z1, sources=y)
    dz1_d=tape.gradient(target=z1,sources=[x,y])
    print("dz1_dx:", dz1_dx)
    print("dz1_dy:", dz1_dy)
    print("dz1_d:",dz1_d)
    print("type of dz1_d:",type(dz1_d))


#tape_test1()



#import matplotlib.pyplot as plt

def tape_test2():
    TRAIN_STEPS=20
    
    # Prepare train data
    train_X = np.linspace(-1, 1, 100)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
    
    print(train_X.shape)
    
    ww=tf2.Variable(initial_value=1.0)
    bb=tf2.Variable(initial_value=1.0)
    
    optimizer=tf2.keras.optimizers.SGD(0.1)
    mse=tf2.keras.losses.MeanSquaredError()
    
    for i in range(TRAIN_STEPS):
        print("epoch:",i)
        #print("w:", w.numpy())
        #print("b:", b.numpy())
        #计算和更新梯度
        with tf2.GradientTape() as tape:
            logit = ww * train_X + bb
            loss=mse(train_Y,logit)
        gradients=tape.gradient(target=loss,sources=[ww,bb])  #计算梯度
        #print("gradients:",gradients)
        #print("zip:\n",list(zip(gradients,[w,b])))
        optimizer.apply_gradients(zip(gradients,[ww,bb]))     #更新梯度
        print("0-w ", gradients[0].numpy(), ww.numpy())
        print("1-b", gradients[1].numpy(), bb.numpy())
        print("loss: ", loss)

    #draw
    '''
    plt.plot(train_X,train_Y,"+")
    plt.plot(train_X,ww * train_X + bb)
    plt.show()
    '''
#tape_test2()



#tape_test3()
