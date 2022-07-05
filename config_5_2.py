#v5.1: state3s_fraud
#v5.2: state refctory
#      default env: game reward only rather than round
#      reward alg changed. fmt[-1, 0.5, 1, 2] is obsolete
#      TD support
'''
         env1  env2   env3 (batch=3)
episode1
episode1   |       -----> ENV
...        |
episoden   V BATCH
BATCH/ENV direction

keep_batch | keep_env |  vs....
----------------------------------
True       |True      | BATCH same, ENV same
----------------------------------
True       |False     | BATCH same, ENV diff
----------------------------------
False      |True      | BATCH same, ENV same  (seldom use, 易混淆)
----------------------------------
False      |False     | BATCH diff, ENV diff
----------------------------------
'''

########################################
agent_config_sets=[]  #1+5
# agent id 0~9999: for env=0
#                        id,     agent_name,    net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([0,     'MC_q',        [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.2,     0.9])
agent_config_sets.append([1,     'MC_q',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.9])
agent_config_sets.append([2,     'MC_q',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.9]) #0.948*2
agent_config_sets.append([3,     'MC_q',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.9])
agent_config_sets.append([4,     'MC_q',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.9])
agent_config_sets.append([5,     'MC_q',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.9])
agent_config_sets.append([6,     'MC_q',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.9])
agent_config_sets.append([7,     'MC_q',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.5])
agent_config_sets.append([8,     'MC_q',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.5])
agent_config_sets.append([9,     'MC_q',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.5])
agent_config_sets.append([10,    'MC_q',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.5])
agent_config_sets.append([11,    'MC_q',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.5])
agent_config_sets.append([12,    'MC_q',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.5])


#                        id,     agent_name,    net-conf,                                                                                 lr,          epsilon, gamma(epsilon meaningless in b)
agent_config_sets.append([100,   'MC_q_b',      [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.0,     0.9])
agent_config_sets.append([101,   'MC_q_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.9])
agent_config_sets.append([102,   'MC_q_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.9]) #0.948*2
agent_config_sets.append([103,   'MC_q_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.9])
agent_config_sets.append([104,   'MC_q_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.9])
agent_config_sets.append([105,   'MC_q_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.9])
agent_config_sets.append([106,   'MC_q_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.9])
agent_config_sets.append([107,   'MC_q_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.5])
agent_config_sets.append([108,   'MC_q_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.5])
agent_config_sets.append([109,   'MC_q_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.5])
agent_config_sets.append([110,   'MC_q_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.5])
agent_config_sets.append([111,   'MC_q_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.5])
agent_config_sets.append([112,   'MC_q_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.5])

                                                                                                     
#                        id,     agent_name,    net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([200,   'MC_q_sm',     [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.2,     0.9])
agent_config_sets.append([201,   'MC_q_sm',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.9])
agent_config_sets.append([202,   'MC_q_sm',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.9]) #0.948*2
agent_config_sets.append([203,   'MC_q_sm',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.9])
agent_config_sets.append([204,   'MC_q_sm',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.9])
agent_config_sets.append([205,   'MC_q_sm',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.9])
agent_config_sets.append([206,   'MC_q_sm',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.9])
agent_config_sets.append([207,   'MC_q_sm',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.5])
agent_config_sets.append([208,   'MC_q_sm',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.5])
agent_config_sets.append([209,   'MC_q_sm',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.5])
agent_config_sets.append([210,   'MC_q_sm',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.5])
agent_config_sets.append([211,   'MC_q_sm',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.5])
agent_config_sets.append([212,   'MC_q_sm',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.5])


                                                                                                     
#                        id,     agent_name,    net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([300,   'MC_q_avg',    [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.2,     0.9])
agent_config_sets.append([301,   'MC_q_avg',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.9])
agent_config_sets.append([302,   'MC_q_avg',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.9]) #0.948*2
agent_config_sets.append([303,   'MC_q_avg',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.9])
agent_config_sets.append([304,   'MC_q_avg',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.9])
agent_config_sets.append([305,   'MC_q_avg',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.9])
agent_config_sets.append([306,   'MC_q_avg',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.9])
agent_config_sets.append([307,   'MC_q_avg',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.5])
agent_config_sets.append([308,   'MC_q_avg',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.5])
agent_config_sets.append([309,   'MC_q_avg',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.5])
agent_config_sets.append([310,   'MC_q_avg',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.5])
agent_config_sets.append([311,   'MC_q_avg',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.5])
agent_config_sets.append([312,   'MC_q_avg',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.5])


                                                                                                     
#                        id,     agent_name,    net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([400,   'MC_pi_b',     [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.9])
agent_config_sets.append([401,   'MC_pi_b',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([402,   'MC_pi_b',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9]) #0.948*2
agent_config_sets.append([403,   'MC_pi_b',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([404,   'MC_pi_b',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([405,   'MC_pi_b',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([406,   'MC_pi_b',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([407,   'MC_pi_b',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([408,   'MC_pi_b',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([409,   'MC_pi_b',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([410,   'MC_pi_b',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([411,   'MC_pi_b',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([412,   'MC_pi_b',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])
                                                                                                   

#                        id,     agent_name,    net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([500,   'MC_pi',       [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.9])
agent_config_sets.append([501,   'MC_pi',       [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([502,   'MC_pi',       [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9]) #0.948*2
agent_config_sets.append([503,   'MC_pi',       [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([504,   'MC_pi',       [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([505,   'MC_pi',       [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([506,   'MC_pi',       [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([507,   'MC_pi',       [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([508,   'MC_pi',       [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([509,   'MC_pi',       [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([510,   'MC_pi',       [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([511,   'MC_pi',       [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([512,   'MC_pi',       [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])
                                                                                                   

#                        id,     agent_name,    net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([600,   'MC_q_f',      [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.2,     0.9])
agent_config_sets.append([601,   'MC_q_f',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.9])
agent_config_sets.append([602,   'MC_q_f',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.9]) #1.227*2
agent_config_sets.append([603,   'MC_q_f',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.9])
agent_config_sets.append([604,   'MC_q_f',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.9])
agent_config_sets.append([605,   'MC_q_f',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.9])
agent_config_sets.append([606,   'MC_q_f',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.9])
agent_config_sets.append([607,   'MC_q_f',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.5])
agent_config_sets.append([608,   'MC_q_f',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.5])
agent_config_sets.append([609,   'MC_q_f',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.5])
agent_config_sets.append([610,   'MC_q_f',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.5])
agent_config_sets.append([611,   'MC_q_f',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.5])
agent_config_sets.append([612,   'MC_q_f',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.5])


#                        id,     agent_name,    net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([700,   'MC_q_f_b',    [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.0,     0.9])
agent_config_sets.append([701,   'MC_q_f_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.9])
agent_config_sets.append([702,   'MC_q_f_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.9]) #1.227*2
agent_config_sets.append([703,   'MC_q_f_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.9])
agent_config_sets.append([704,   'MC_q_f_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.9])
agent_config_sets.append([705,   'MC_q_f_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.9])
agent_config_sets.append([706,   'MC_q_f_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.9])
agent_config_sets.append([707,   'MC_q_f_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.5])
agent_config_sets.append([708,   'MC_q_f_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.5])
agent_config_sets.append([709,   'MC_q_f_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.5])
agent_config_sets.append([710,   'MC_q_f_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.5])
agent_config_sets.append([711,   'MC_q_f_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.5])
agent_config_sets.append([712,   'MC_q_f_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.5])


#                        id,     agent_name,    net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([800,   'MC_pi_f',     [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.9])
agent_config_sets.append([801,   'MC_pi_f',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([802,   'MC_pi_f',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9]) #1.227*2
agent_config_sets.append([803,   'MC_pi_f',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([804,   'MC_pi_f',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([805,   'MC_pi_f',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([806,   'MC_pi_f',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([807,   'MC_pi_f',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([808,   'MC_pi_f',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([809,   'MC_pi_f',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([810,   'MC_pi_f',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([811,   'MC_pi_f',     [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([812,   'MC_pi_f',     [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])


#                        id,     agent_name,    net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([900,   'MC_pi_f_b',   [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.9])
agent_config_sets.append([901,   'MC_pi_f_b',   [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([902,   'MC_pi_f_b',   [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9]) #1.227*2
agent_config_sets.append([903,   'MC_pi_f_b',   [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([904,   'MC_pi_f_b',   [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([905,   'MC_pi_f_b',   [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([906,   'MC_pi_f_b',   [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([907,   'MC_pi_f_b',   [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([908,   'MC_pi_f_b',   [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([909,   'MC_pi_f_b',   [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([910,   'MC_pi_f_b',   [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([911,   'MC_pi_f_b',   [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([912,   'MC_pi_f_b',   [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])


#                        id,     agent_name,    net-conf(input net),                                                                                                                                                         (output net)                                                          lr,          epsilon, gamma
agent_config_sets.append([1000,  'C_MC_q',      {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])
agent_config_sets.append([1001,  'C_MC_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])
agent_config_sets.append([1002,  'C_MC_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])  #53.946*2
agent_config_sets.append([1003,  'C_MC_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.9])
agent_config_sets.append([1004,  'C_MC_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.9])
agent_config_sets.append([1005,  'C_MC_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.9])
agent_config_sets.append([1006,  'C_MC_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.9])
agent_config_sets.append([1007,  'C_MC_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.5])
agent_config_sets.append([1008,  'C_MC_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.5])
agent_config_sets.append([1009,  'C_MC_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.5])
agent_config_sets.append([1010,  'C_MC_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.5])
agent_config_sets.append([1011,  'C_MC_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.5])
agent_config_sets.append([1012,  'C_MC_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.5])



#                        id,     agent_name,    net-conf(input net),                                                                                                                                                         (output net)                                                          lr,          epsilon, gamma
agent_config_sets.append([1100,  'C_MC_q_b',    {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(2,2)}},  0.001,       0.2,     0.9])
agent_config_sets.append([1101,  'C_MC_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])
agent_config_sets.append([1102,  'C_MC_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])  #53.946*2
agent_config_sets.append([1103,  'C_MC_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.9])
agent_config_sets.append([1104,  'C_MC_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.9])
agent_config_sets.append([1105,  'C_MC_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.9])
agent_config_sets.append([1106,  'C_MC_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.9])
agent_config_sets.append([1107,  'C_MC_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.5])
agent_config_sets.append([1108,  'C_MC_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.5])
agent_config_sets.append([1109,  'C_MC_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.5])
agent_config_sets.append([1110,  'C_MC_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.5])
agent_config_sets.append([1111,  'C_MC_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.5])
agent_config_sets.append([1112,  'C_MC_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.5])


#                        id,     agent_name,      net-conf(input net),                                                                            (residual_net)                                                                                                                                                                         (policy net)                                                            (v_net)                                                            lr,           epsilon, gamma
agent_config_sets.append([1200,  'Res_MC_pi',     {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,        0.0,     0.9])
agent_config_sets.append([1201,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([1202,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])   #38.576
agent_config_sets.append([1203,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1204,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1205,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1206,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1207,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1208,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1209,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1210,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1211,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([1212,  'Res_MC_pi',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])


#                        id,     agent_name,      net-conf(input net),                                                                            (residual_net)                                                                                                                                                                         (policy net)                                                            (v_net)                                                            lr,           epsilon, gamma
agent_config_sets.append([1300,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,        0.0,     0.9])
agent_config_sets.append([1301,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([1302,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])   #7.392
agent_config_sets.append([1303,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1304,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1305,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1306,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1307,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1308,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1309,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1310,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1311,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([1312,  'Res_MC_pi_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])


#                        id,     agent_name,    net-conf(input net),                                                                                                                                                               (residual_net, empty)                                                      (policy net)                                                            (v_net)                                                              lr,           epsilon, gamma
agent_config_sets.append([1400,  'C_MC_pi',     {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                              'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(2,2)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(2,2)}},    0.001,        0.0,     0.9])
agent_config_sets.append([1401,  'C_MC_pi',     {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([1402,  'C_MC_pi',     {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.9])    #8.142
agent_config_sets.append([1403,  'C_MC_pi',     {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1404,  'C_MC_pi',     {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1405,  'C_MC_pi',     {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1406,  'C_MC_pi',     {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1407,  'C_MC_pi',     {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1408,  'C_MC_pi',     {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1409,  'C_MC_pi',     {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1410,  'C_MC_pi',     {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1411,  'C_MC_pi',     {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([1412,  'C_MC_pi',     {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.5])

#                        id,     agent_name,    net-conf(input net),                                                                                                                                                              (residual_net, empty)                                                      (policy net)                                                            (v_net)                                                               lr,           epsilon, gamma
agent_config_sets.append([1500,  'C_MC_pi_b',   {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                              'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(2,2)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(2,2)}},    0.001,        0.0,     0.9])
agent_config_sets.append([1501,  'C_MC_pi_b',   {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([1502,  'C_MC_pi_b',   {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.9]) #70.747
agent_config_sets.append([1503,  'C_MC_pi_b',   {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1504,  'C_MC_pi_b',   {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([1505,  'C_MC_pi_b',   {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1506,  'C_MC_pi_b',   {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([1507,  'C_MC_pi_b',   {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1508,  'C_MC_pi_b',   {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([1509,  'C_MC_pi_b',   {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1510,  'C_MC_pi_b',   {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([1511,  'C_MC_pi_b',   {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                                'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([1512,  'C_MC_pi_b',   {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},           'residual_net':{'conv_filters':[], 'kernal_sizes':[],  'strides':[]},      'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,1)},    'v_net':{'conv_filter': 64,    'kernal_size':2, 'stride':(1,1)}},    0.000001,     0.0,     0.5])

#                        id,     agent_name,      net-conf(input net),                                                                              (residual_net)                                                                                                                                                                          (output net)                                                            lr,      epsilon, gamma
agent_config_sets.append([1600,  'Res_MC_q',     {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.2,     0.9])
agent_config_sets.append([1601,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.2,     0.9])
agent_config_sets.append([1602,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.2,     0.9]) #38.208*2
agent_config_sets.append([1603,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.2,     0.9])
agent_config_sets.append([1604,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.2,     0.9])
agent_config_sets.append([1605,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.2,     0.9])
agent_config_sets.append([1606,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.2,     0.9])
agent_config_sets.append([1607,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.2,     0.5])
agent_config_sets.append([1608,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.2,     0.5])
agent_config_sets.append([1609,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.2,     0.5])
agent_config_sets.append([1610,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.2,     0.5])
agent_config_sets.append([1611,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.2,     0.5])
agent_config_sets.append([1612,  'Res_MC_q',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.2,     0.5])


#                        id,     agent_name,      net-conf(input net),                                                                               (residual_net)                                                                                                                                                                         (output net)                                                            lr,      epsilon, gamma
agent_config_sets.append([1700,  'Res_MC_q_b',   {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.2,     0.9])
agent_config_sets.append([1701,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.0,     0.9])
agent_config_sets.append([1702,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.0,     0.9])   #38.028*2
agent_config_sets.append([1703,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.0,     0.9])
agent_config_sets.append([1704,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.0,     0.9])
agent_config_sets.append([1705,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.0,     0.9])
agent_config_sets.append([1706,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.0,     0.9])
agent_config_sets.append([1707,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.0,     0.5])
agent_config_sets.append([1708,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,   0.0,     0.5])
agent_config_sets.append([1709,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.0,     0.5])
agent_config_sets.append([1710,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,  0.0,     0.5])
agent_config_sets.append([1711,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                        'strides':[[(1,1), (1,1)],]*4},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.0,     0.5])
agent_config_sets.append([1712,  'Res_MC_q_b',   {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},           'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'output_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001, 0.0,     0.5])


#                        id,     agent_name,       net-conf(input net),                                                                                                                                                        (output net)                                                          lr,          epsilon, gamma
agent_config_sets.append([1800,  'C_MC_q_f',      {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])
agent_config_sets.append([1801,  'C_MC_q_f',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])
agent_config_sets.append([1802,  'C_MC_q_f',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])    #54.139*2
agent_config_sets.append([1803,  'C_MC_q_f',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.9])
agent_config_sets.append([1804,  'C_MC_q_f',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.9])
agent_config_sets.append([1805,  'C_MC_q_f',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.9])
agent_config_sets.append([1806,  'C_MC_q_f',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.9])
agent_config_sets.append([1807,  'C_MC_q_f',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.5])
agent_config_sets.append([1808,  'C_MC_q_f',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.5])
agent_config_sets.append([1809,  'C_MC_q_f',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.5])
agent_config_sets.append([1810,  'C_MC_q_f',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.5])
agent_config_sets.append([1811,  'C_MC_q_f',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.5])
agent_config_sets.append([1812,  'C_MC_q_f',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.5])



#                        id,     agent_name,         net-conf(input net),                                                                                                                                                        (output net)                                                          lr,          epsilon, gamma
agent_config_sets.append([1900,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])
agent_config_sets.append([1901,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])
agent_config_sets.append([1902,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])  #54.139*2
agent_config_sets.append([1903,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.9])
agent_config_sets.append([1904,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.9])
agent_config_sets.append([1905,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.9])
agent_config_sets.append([1906,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.9])
agent_config_sets.append([1907,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.5])
agent_config_sets.append([1908,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.5])
agent_config_sets.append([1909,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.5])
agent_config_sets.append([1910,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.5])
agent_config_sets.append([1911,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.5])
agent_config_sets.append([1912,  'C_MC_q_f_b',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.5])


#                        id,     agent_name,         net-conf(input net),                                                                          (residual_net)                                                                                                                                                                          (policy net)                                                            (v_net)                                                            lr,           epsilon, gamma
agent_config_sets.append([2000,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,        0.0,     0.9])
agent_config_sets.append([2001,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([2002,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])  #38.702
agent_config_sets.append([2003,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([2004,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([2005,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([2006,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([2007,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([2008,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([2009,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([2010,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([2011,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([2012,  'Res_MC_pi_f',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                     
#                        id,     agent_name,           net-conf(input net),                                                                        (residual_net)                                                                                                                                                                          (policy net)                                                            (v_net)                                                            lr,           epsilon, gamma
agent_config_sets.append([2100,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [32],         'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,        0.0,     0.9])
agent_config_sets.append([2101,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([2102,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])  #38.702
agent_config_sets.append([2103,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([2104,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([2105,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([2106,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([2107,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([2108,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([2109,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([2110,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([2111,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([2112,  'Res_MC_pi_f_b',     {'input_net':{'conv_filters': [256],        'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])


#                        id,       agent_name,      net-conf,                                                                                  lr,          epsilon, gamma
agent_config_sets.append([2200,    'TD_q',          [[64, 0.2],[16, 0.2]],                                                                     0.001,       0.2,     0.9])
agent_config_sets.append([2201,    'TD_q',          [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                         0.001,       0.2,     0.9])
agent_config_sets.append([2202,    'TD_q',          [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],     0.001,       0.2,     0.9])   #0.948 *2
agent_config_sets.append([2203,    'TD_q',          [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                         0.0001,      0.2,     0.9])
agent_config_sets.append([2204,    'TD_q',          [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],     0.0001,      0.2,     0.9])
agent_config_sets.append([2205,    'TD_q',          [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                         0.00001,     0.2,     0.9])
agent_config_sets.append([2206,    'TD_q',          [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],     0.00001,     0.2,     0.9])
agent_config_sets.append([2207,    'TD_q',          [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                         0.001,       0.2,     0.5])
agent_config_sets.append([2208,    'TD_q',          [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],     0.001,       0.2,     0.5])
agent_config_sets.append([2209,    'TD_q',          [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                         0.0001,      0.2,     0.5])
agent_config_sets.append([2210,    'TD_q',          [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],     0.0001,      0.2,     0.5])
agent_config_sets.append([2211,    'TD_q',          [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                         0.00001,     0.2,     0.5])
agent_config_sets.append([2212,    'TD_q',          [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],     0.00001,     0.2,     0.5])

#                        id,       agent_name,      net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([2300,    'TD_q_b',        [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.0,     0.9])
agent_config_sets.append([2301,    'TD_q_b',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.9])
agent_config_sets.append([2302,    'TD_q_b',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.9]) #0.948*2
agent_config_sets.append([2303,    'TD_q_b',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.9])
agent_config_sets.append([2304,    'TD_q_b',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.9])
agent_config_sets.append([2305,    'TD_q_b',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.9])
agent_config_sets.append([2306,    'TD_q_b',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.9])
agent_config_sets.append([2307,    'TD_q_b',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.5])
agent_config_sets.append([2308,    'TD_q_b',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.5])
agent_config_sets.append([2309,    'TD_q_b',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.5])
agent_config_sets.append([2310,    'TD_q_b',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.5])
agent_config_sets.append([2311,    'TD_q_b',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.5])
agent_config_sets.append([2312,    'TD_q_b',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.5])




#                        id,       agent_name,      net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([2400,    'TD_q_f',        [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.2,     0.9])
agent_config_sets.append([2401,    'TD_q_f',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.9])
agent_config_sets.append([2402,    'TD_q_f',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.9]) #1.227*2
agent_config_sets.append([2403,    'TD_q_f',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.9])
agent_config_sets.append([2404,    'TD_q_f',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.9])
agent_config_sets.append([2405,    'TD_q_f',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.9])
agent_config_sets.append([2406,    'TD_q_f',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.9])
agent_config_sets.append([2407,    'TD_q_f',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.2,     0.5])
agent_config_sets.append([2408,    'TD_q_f',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.2,     0.5])
agent_config_sets.append([2409,    'TD_q_f',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.2,     0.5])
agent_config_sets.append([2410,    'TD_q_f',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.2,     0.5])
agent_config_sets.append([2411,    'TD_q_f',        [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.2,     0.5])
agent_config_sets.append([2412,    'TD_q_f',        [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.2,     0.5])

#                        id,       agent_name,      net-conf,                                                                                 lr,          epsilon, gamma
agent_config_sets.append([2500,    'TD_q_f_b',      [[64, 0.2],[16, 0.2]],                                                                    0.001,       0.0,     0.9])
agent_config_sets.append([2501,    'TD_q_f_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.9])
agent_config_sets.append([2502,    'TD_q_f_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.9])   #1.227*2
agent_config_sets.append([2503,    'TD_q_f_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.9])
agent_config_sets.append([2504,    'TD_q_f_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.9])
agent_config_sets.append([2505,    'TD_q_f_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.9])
agent_config_sets.append([2506,    'TD_q_f_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.9])
agent_config_sets.append([2507,    'TD_q_f_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.001,       0.0,     0.5])
agent_config_sets.append([2508,    'TD_q_f_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.001,       0.0,     0.5])
agent_config_sets.append([2509,    'TD_q_f_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,      0.0,     0.5])
agent_config_sets.append([2510,    'TD_q_f_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,      0.0,     0.5])
agent_config_sets.append([2511,    'TD_q_f_b',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,     0.0,     0.5])
agent_config_sets.append([2512,    'TD_q_f_b',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,     0.0,     0.5])



#                        id,       agent_name,      net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([2600,   'MC_pi_acc',      [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.5])
agent_config_sets.append([2601,   'MC_pi_acc',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([2602,   'MC_pi_acc',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9])    #0.948*2
agent_config_sets.append([2603,   'MC_pi_acc',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([2604,   'MC_pi_acc',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([2605,   'MC_pi_acc',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([2606,   'MC_pi_acc',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([2607,   'MC_pi_acc',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([2608,   'MC_pi_acc',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([2609,   'MC_pi_acc',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([2610,   'MC_pi_acc',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([2611,   'MC_pi_acc',      [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([2612,   'MC_pi_acc',      [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])



#                        id,       agent_name,      net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([2700,   'MC_pi_acc_b',    [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.5])
agent_config_sets.append([2701,   'MC_pi_acc_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([2702,   'MC_pi_acc_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9])  #0.948*2
agent_config_sets.append([2703,   'MC_pi_acc_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([2704,   'MC_pi_acc_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([2705,   'MC_pi_acc_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([2706,   'MC_pi_acc_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([2707,   'MC_pi_acc_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([2708,   'MC_pi_acc_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([2709,   'MC_pi_acc_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([2710,   'MC_pi_acc_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([2711,   'MC_pi_acc_b',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([2712,   'MC_pi_acc_b',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])



#                        id,       agent_name,      net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([2800,   'MC_pi_acc_f',    [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.5])
agent_config_sets.append([2801,   'MC_pi_acc_f',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([2802,   'MC_pi_acc_f',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9])  #1.227*2
agent_config_sets.append([2803,   'MC_pi_acc_f',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([2804,   'MC_pi_acc_f',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([2805,   'MC_pi_acc_f',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([2806,   'MC_pi_acc_f',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([2807,   'MC_pi_acc_f',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([2808,   'MC_pi_acc_f',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([2809,   'MC_pi_acc_f',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([2810,   'MC_pi_acc_f',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([2811,   'MC_pi_acc_f',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([2812,   'MC_pi_acc_f',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])

#                        id,       agent_name,      net-conf,                                                                                 lr,         epsilon, gamma
agent_config_sets.append([2900,   'MC_pi_acc_f_b',  [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     0.5])
agent_config_sets.append([2901,   'MC_pi_acc_f_b',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.9])
agent_config_sets.append([2902,   'MC_pi_acc_f_b',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.9])    #1.227*2
agent_config_sets.append([2903,   'MC_pi_acc_f_b',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.9])
agent_config_sets.append([2904,   'MC_pi_acc_f_b',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.9])
agent_config_sets.append([2905,   'MC_pi_acc_f_b',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.9])
agent_config_sets.append([2906,   'MC_pi_acc_f_b',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.9])
agent_config_sets.append([2907,   'MC_pi_acc_f_b',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     0.5])
agent_config_sets.append([2908,   'MC_pi_acc_f_b',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     0.5])
agent_config_sets.append([2909,   'MC_pi_acc_f_b',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     0.5])
agent_config_sets.append([2910,   'MC_pi_acc_f_b',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     0.5])
agent_config_sets.append([2911,   'MC_pi_acc_f_b',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     0.5])
agent_config_sets.append([2912,   'MC_pi_acc_f_b',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     0.5])


#                        id,       agent_name,      net-conf,                                                                                 lr,         epsilon, gamma,  lamda-a, lamda-c
agent_config_sets.append([3000,   'MC_pi_acc_e',    [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     [0.5,   0.9,     0.9]])
agent_config_sets.append([3001,   'MC_pi_acc_e',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3002,   'MC_pi_acc_e',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     [0.9,   0.9,     0.9]])    #0.948*2
agent_config_sets.append([3003,   'MC_pi_acc_e',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3004,   'MC_pi_acc_e',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3005,   'MC_pi_acc_e',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3006,   'MC_pi_acc_e',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3007,   'MC_pi_acc_e',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3008,   'MC_pi_acc_e',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3009,   'MC_pi_acc_e',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3010,   'MC_pi_acc_e',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3011,   'MC_pi_acc_e',    [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3012,   'MC_pi_acc_e',    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     [0.5,   0.5,     0.5]])


#                        id,       agent_name,      net-conf,                                                                                 lr,         epsilon, gamma,  lamda-a, lamda-c
agent_config_sets.append([3100,   'MC_pi_acc_e_f',  [[64, 0.2],[16, 0.2]],                                                                    0.001,      0.0,     [0.5,   0.9,     0.9]])
agent_config_sets.append([3101,   'MC_pi_acc_e_f',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3102,   'MC_pi_acc_e_f',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     [0.9,   0.9,     0.9]])  #1.227*2
agent_config_sets.append([3103,   'MC_pi_acc_e_f',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3104,   'MC_pi_acc_e_f',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3105,   'MC_pi_acc_e_f',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3106,   'MC_pi_acc_e_f',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     [0.9,   0.9,     0.9]])
agent_config_sets.append([3107,   'MC_pi_acc_e_f',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.0001,     0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3108,   'MC_pi_acc_e_f',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.0001,     0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3109,   'MC_pi_acc_e_f',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.00001,    0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3110,   'MC_pi_acc_e_f',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.00001,    0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3111,   'MC_pi_acc_e_f',  [[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        0.000001,   0.0,     [0.5,   0.5,     0.5]])
agent_config_sets.append([3112,   'MC_pi_acc_e_f',  [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    0.000001,   0.0,     [0.5,   0.5,     0.5]])



#                        id,     agent_name,     net-conf(input net),                                                                                                                                                        (output net)                                                          lr,          epsilon, gamma
agent_config_sets.append([3200,  'C_TD_q',      {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])
agent_config_sets.append([3201,  'C_TD_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])
agent_config_sets.append([3202,  'C_TD_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.9])  #53.946*2
agent_config_sets.append([3203,  'C_TD_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.9])
agent_config_sets.append([3204,  'C_TD_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.9])
agent_config_sets.append([3205,  'C_TD_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.9])
agent_config_sets.append([3216,  'C_TD_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.9])
agent_config_sets.append([3217,  'C_TD_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.5])
agent_config_sets.append([3218,  'C_TD_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.2,     0.5])
agent_config_sets.append([3219,  'C_TD_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.5])
agent_config_sets.append([3210,  'C_TD_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.2,     0.5])
agent_config_sets.append([3211,  'C_TD_q',      {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.5])
agent_config_sets.append([3212,  'C_TD_q',      {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.2,     0.5])




#                        id,     agent_name,     net-conf(input net),                                                                                                                                                        (output net)                                                          lr,          epsilon, gamma
agent_config_sets.append([3300,  'C_TD_q_b',    {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])
agent_config_sets.append([3301,  'C_TD_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])
agent_config_sets.append([3302,  'C_TD_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.9])    #53.946*2
agent_config_sets.append([3303,  'C_TD_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.9])
agent_config_sets.append([3304,  'C_TD_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.9])
agent_config_sets.append([3305,  'C_TD_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.9])
agent_config_sets.append([3306,  'C_TD_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.9])
agent_config_sets.append([3307,  'C_TD_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.5])
agent_config_sets.append([3308,  'C_TD_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.001,       0.0,     0.5])
agent_config_sets.append([3309,  'C_TD_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.5])
agent_config_sets.append([3310,  'C_TD_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.0001,      0.0,     0.5])
agent_config_sets.append([3311,  'C_TD_q_b',    {'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.5])
agent_config_sets.append([3312,  'C_TD_q_b',    {'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},  0.00001,     0.0,     0.5])


#                        id,      agent_name,          net-conf(input net),                                                                          (residual_net)                                                                                                                                                                          (policy net)                                                            (v_net)                                                            lr,           epsilon, gamma
agent_config_sets.append([3400,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,        0.0,     0.5])
agent_config_sets.append([3401,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([3402,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])   #38.576
agent_config_sets.append([3403,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([3404,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([3405,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([3406,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([3407,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([3408,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([3409,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([3410,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([3411,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([3412,  'Res_MC_pi_acc',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])




#                        id,      agent_name,            net-conf(input net),                                                                          (residual_net)                                                                                                                                                                          (policy net)                                                            (v_net)                                                            lr,           epsilon, gamma
agent_config_sets.append([3500,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [32],           'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[32, 32],]*2,            'kernal_sizes':[[2,2], [2,2]],                                                     'strides':[[(1,1), (1,1)],]*2},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.001,        0.0,     0.5])
agent_config_sets.append([3501,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9])
agent_config_sets.append([3502,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.9]) #38.576
agent_config_sets.append([3503,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([3504,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.9])
agent_config_sets.append([3505,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([3506,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.9])
agent_config_sets.append([3507,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([3508,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.0001,       0.0,     0.5])
agent_config_sets.append([3509,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([3510,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.00001,      0.0,     0.5])
agent_config_sets.append([3511,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*4,          'kernal_sizes':[[3,3], [3,3], [2,2], [2,2]],                                       'strides':[[(1,1), (1,1)],]*4},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])
agent_config_sets.append([3512,  'Res_MC_pi_acc_b',     {'input_net':{'conv_filters': [256],          'kernal_sizes':[3],    'strides':[(1,3)]},       'residual_net':{'conv_filters':[[256, 256],]*7,          'kernal_sizes':[[3,3], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2]],                  'strides':[[(1,1), (1,1)],]*7},             'policy_net':{'conv_filter': 64,   'kernal_size':2, 'stride':(1,3)},    'v_net':{'conv_filter': 64,  'kernal_size':3, 'stride':(1,3)}},    0.000001,     0.0,     0.5])


#                        id,      agent_name,    net-conf,                                                                                  guess net                                                           lr,          epsilon, gamma
agent_config_sets.append([3600,   'MC_q_gd',     [[[64, 0.2],[16, 0.2]],                                                                    [[64, 0.2],[16, 0.2]]],                                             0.001,       0.2,     0.9])
agent_config_sets.append([3601,   'MC_q_gd',     [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.001,       0.2,     0.9])
agent_config_sets.append([3602,   'MC_q_gd',     [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.001,       0.2,     0.9])    #1.227*3
agent_config_sets.append([3603,   'MC_q_gd',     [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.0001,      0.2,     0.9])
agent_config_sets.append([3604,   'MC_q_gd',     [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.0001,      0.2,     0.9])
agent_config_sets.append([3605,   'MC_q_gd',     [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.00001,     0.2,     0.9])
agent_config_sets.append([3606,   'MC_q_gd',     [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.00001,     0.2,     0.9])
agent_config_sets.append([3607,   'MC_q_gd',     [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.001,       0.2,     0.5])
agent_config_sets.append([3608,   'MC_q_gd',     [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.001,       0.2,     0.5])
agent_config_sets.append([3609,   'MC_q_gd',     [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.0001,      0.2,     0.5])
agent_config_sets.append([3610,   'MC_q_gd',     [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.0001,      0.2,     0.5])
agent_config_sets.append([3611,   'MC_q_gd',     [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.00001,     0.2,     0.5])
agent_config_sets.append([3612,   'MC_q_gd',     [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.00001,     0.2,     0.5])


#                         id,      agent_name,      net-conf,                                                                               guess net                                                           lr,          epsilon, gamma
agent_config_sets.append([3700,   'MC_q_gd_b',   [[[64, 0.2],[16, 0.2]],                                                                    [[64, 0.2],[16, 0.2]]],                                             0.001,       0.2,     0.9])
agent_config_sets.append([3701,   'MC_q_gd_b',   [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.001,       0.0,     0.9])
agent_config_sets.append([3702,   'MC_q_gd_b',   [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.001,       0.0,     0.9])  #1.227*3
agent_config_sets.append([3703,   'MC_q_gd_b',   [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.0001,      0.0,     0.9])
agent_config_sets.append([3704,   'MC_q_gd_b',   [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.0001,      0.0,     0.9])
agent_config_sets.append([3705,   'MC_q_gd_b',   [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.00001,     0.0,     0.9])
agent_config_sets.append([3706,   'MC_q_gd_b',   [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.00001,     0.0,     0.9])
agent_config_sets.append([3707,   'MC_q_gd_b',   [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.001,       0.0,     0.5])
agent_config_sets.append([3708,   'MC_q_gd_b',   [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.001,       0.0,     0.5])
agent_config_sets.append([3709,   'MC_q_gd_b',   [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.0001,      0.0,     0.5])
agent_config_sets.append([3710,   'MC_q_gd_b',   [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.0001,      0.0,     0.5])
agent_config_sets.append([3711,   'MC_q_gd_b',   [[[1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3]],                                        [[1024, 0.3], [512, 0.3], [256, 0.3]]],                             0.00001,     0.0,     0.5])
agent_config_sets.append([3712,   'MC_q_gd_b',   [[[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3], [128, 0.3], [64, 0.3]],    [[512, 0.3], [2048, 0.3], [1024, 0.3], [512, 0.3], [256, 0.3]]],    0.00001,     0.0,     0.5])


#                         id,     agent_name,     net-conf(input net),                                                                                                                                                          (output net)                                                                          guess net                                                                                                                                                                                                                                             lr,          epsilon, gamma
agent_config_sets.append([3800,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.2,     0.9])
agent_config_sets.append([3801,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.2,     0.9])
agent_config_sets.append([3802,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.2,     0.9])  #54.139*3
agent_config_sets.append([3803,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.2,     0.9])
agent_config_sets.append([3804,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.2,     0.9])
agent_config_sets.append([3805,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.2,     0.9])
agent_config_sets.append([3806,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.2,     0.9])
agent_config_sets.append([3807,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.2,     0.5])
agent_config_sets.append([3808,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.2,     0.5])
agent_config_sets.append([3809,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.2,     0.5])
agent_config_sets.append([3810,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.2,     0.5])
agent_config_sets.append([3811,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.2,     0.5])
agent_config_sets.append([3812,  'C_MC_q_gc',     [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.2,     0.5])



#                         id,     agent_name,      net-conf(input net),                                                                                                                                                         (output net)                                                                          guess net                                                                                                                                                                                                                                             lr,          epsilon, gamma
agent_config_sets.append([3900,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32],                              'kernal_sizes':[3, 2],                'strides':[(1,2), (1,2)]},                                        'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.2,     0.9])
agent_config_sets.append([3901,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.0,     0.9])
agent_config_sets.append([3902,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.0,     0.9]) #54.129*3
agent_config_sets.append([3903,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.0,     0.9])
agent_config_sets.append([3904,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.0,     0.9])
agent_config_sets.append([3905,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.0,     0.9])
agent_config_sets.append([3906,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.0,     0.9])
agent_config_sets.append([3907,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.0,     0.5])
agent_config_sets.append([3908,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.001,       0.0,     0.5])
agent_config_sets.append([3909,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.0,     0.5])
agent_config_sets.append([3910,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.0001,      0.0,     0.5])
agent_config_sets.append([3911,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [32, 64, 128, 256],                    'kernal_sizes':[3, 3, 2, 2],          'strides':[(1,2), (1,2), (1,2), (2,2)]},                          'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [64, 128, 256],                        'kernal_sizes':[3, 2, 2],             'strides':[(1,3), (1,2), (1,2)]},                                 'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.0,     0.5])
agent_config_sets.append([3912,  'C_MC_q_gc_b',   [{'input_net':{'conv_filters': [8, 16, 32, 64, 128, 256, 512],        'kernal_sizes':[3, 3, 3, 3, 2, 2, 2], 'strides':[(1,2), (1,2), (1,2), (2,2), (1,1), (1,1), (1,1)]},     'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},                 {'input_net':{'conv_filters': [16, 32, 64, 128, 256],                'kernal_sizes':[3, 3, 2, 2, 2],       'strides':[(1,2), (1,2), (1,2), (2,2), (1,1)]},                   'output_net':{'conv_filter': 256, 'kernal_size':2, 'stride':(1,1)}}],     0.00001,     0.0,     0.5])


##################################
# agent id 10000~20000: for env=2. id(env=2) = id(env=0)+10000
# . . .


                                                             
########################################
env_config_sets=[] #1+2 #
#                       id,    play_reward_format, keep_env
env_config_sets.append([0,     [-1, 0.5, 1, 2],    False])   #分round给reward不好. [loss, round_winner, scored/50=(5~40)/50=0.1~0.8), game_winner]
env_config_sets.append([1,     [-1, 0.5, 1, 2],    True ])
env_config_sets.append([2,     [ 0, 0,   0, 1],    False])   #只有game winner才能得分， AlphaZero， Backgammon both reward in this way
env_config_sets.append([3,     [ 0, 0,   0, 1],    True ])
env_config_sets.append([4,     [-1, 0.1, 1, 1],    False])   #[loss, round_winner, scored/50=(5~40)/50=0.1~0.8), game_winner]
env_config_sets.append([5,     [-1, 0.1, 1, 1],    True ])

########################################
game_config_sets=[]  #1+9. 'id' is index of -p<> and h5 file. id<100 is for debug
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,  keep_batch, demos, flag_4in1
game_config_sets.append([1,     0,      0,       1,       2,       3,       50,    500,    False,      0,     False])
game_config_sets.append([2,     0,      4,       5,       6,       7,       50,    500,    False,      0,     False])
game_config_sets.append([3,     0,      2,       3,       2,       3,       50,    3000,   False,      0,     False])
game_config_sets.append([4,     0,      4,       5,       4,       5,       50,    3000,   False,      0,     False])
game_config_sets.append([5,     0,      1,       5,       1,       5,       1,     500000, False,      0,     True])  #=game1444
game_config_sets.append([6,     2,      3604,    3605,    3603,    3607,    100,   300,    False,      0,     False])
game_config_sets.append([7,     2,      602,     1002,    1002,    1002,    100,   30000,  False,      0,     True])
game_config_sets.append([8,     2,      702,     3802,    3802,    3802,    100,   30000,  False,      0,     True])
game_config_sets.append([9,     2,      802,     1000,    1000,    1000,    100,   30000,  False,      0,     True])
game_config_sets.append([10,    2,      3400,    900,     2700,    2900,    50,    200,    False,      1,     True])
game_config_sets.append([11,    2,      1002,    700,     2300,    2500,    100,   30000,  False,      0,     True])
game_config_sets.append([12,    2,      2202,    1900,    1700,    3300,    100,   30000,  False,      0,     True])
game_config_sets.append([13,    2,      3902,    1500,    2100,    3500,    100,   3000,   False,      0,     True])


########################################
# id 700 - 799 are verifying the biggest net size is working or not, on GPU only
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games, keep_batch, demos, flag_4in1
game_config_sets.append([700,   3,      2,       102,     2,       102,     100,   1000,  True,       1,     False])
game_config_sets.append([701,   2,      402,     502,     402,     502,     100,   1000,  True,       1,     False])
game_config_sets.append([702,   4,      802,     902,     802,     902,     100,   1000,  False,      1,     False])
game_config_sets.append([703,   5,      1002,    1102,    1002,    1102,    100,   1000,  False,      1,     False])
game_config_sets.append([704,   1,      1402,    1502,    1402,    1502,    100,   1000,  False,      1,     False])
game_config_sets.append([705,   0,      1802,    1902,    1802,    1902,    100,   1000,  True,       1,     False])
game_config_sets.append([706,   2,      2302,    2402,    2302,    2402,    100,   1000,  False,      1,     False])
game_config_sets.append([707,   2,      2802,    2902,    2802,    2902,    100,   1000,  False,      1,     False])
game_config_sets.append([708,   2,      3202,    3302,    3202,    3302,    100,   1000,  False,      1,     False])
game_config_sets.append([709,   2,      3602,    3702,    3602,    3702,    100,   1000,  False,      1,     False])
game_config_sets.append([710,   3,      202,     302,     202,     302,     100,   1000,  True,       1,     False])
game_config_sets.append([711,   2,      602,     702,     602,     702,     100,   1000,  True,       1,     False])
game_config_sets.append([712,   5,      1202,    1302,    1202,    1302,    100,   1000,  False,      1,     False])
game_config_sets.append([713,   1,      1602,    1702,    1602,    1702,    100,   1000,  False,      1,     False])
game_config_sets.append([714,   0,      2002,    2102,    2002,    2102,    100,   1000,  True,       1,     False])
game_config_sets.append([715,   2,      2602,    2702,    2602,    2702,    100,   1000,  False,      1,     False])
game_config_sets.append([716,   2,      3002,    3102,    3002,    3102,    100,   1000,  False,      1,     False])
game_config_sets.append([717,   2,      3402,    3502,    3402,    3502,    100,   1000,  False,      1,     False])
game_config_sets.append([718,   2,      3802,    3902,    3802,    3902,    100,   1000,  False,      1,     False])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games, keep_batch, demos, flag_4in1
game_config_sets.append([730,   2,      2,       0,       0,       0,       100,   1000,  True,       0,     True])
game_config_sets.append([731,   2,      102,     100,     100,     100,     100,   1000,  True,       0,     True])
game_config_sets.append([732,   3,      202,     400,     400,     400,     100,   1000,  True,       0,     True])
game_config_sets.append([733,   3,      302,     600,     600,     600,     100,   1000,  False,      0,     True])
game_config_sets.append([734,   3,      402,     900,     900,     900,     100,   1000,  False,      0,     True])
game_config_sets.append([735,   4,      502,     1000,    1000,    1000,    100,   1000,  True,       0,     True])
game_config_sets.append([736,   4,      602,     1200,    1200,    1200,    100,   1000,  False,      0,     True])
game_config_sets.append([737,   5,      702,     1500,    1500,    1500,    100,   1000,  True,       0,     True])
game_config_sets.append([738,   5,      802,     1700,    1700,    1700,    100,   1000,  False,      0,     True])
game_config_sets.append([739,   2,      902,     1800,    1800,    1800,    100,   1000,  False,      0,     True])
game_config_sets.append([740,   2,      1002,    0,       0,       0,       100,   1000,  True,       0,     True])
game_config_sets.append([741,   2,      1102,    100,     100,     100,     100,   1000,  True,       0,     True])
game_config_sets.append([742,   3,      1202,    400,     400,     400,     100,   1000,  True,       0,     True])
game_config_sets.append([743,   3,      1302,    600,     600,     600,     100,   1000,  False,      0,     True])
game_config_sets.append([744,   3,      1402,    900,     900,     900,     100,   1000,  False,      0,     True])
game_config_sets.append([745,   4,      1502,    1000,    1000,    1000,    100,   1000,  True,       0,     True])
game_config_sets.append([746,   4,      1602,    1200,    1200,    1200,    100,   1000,  False,      0,     True])
game_config_sets.append([747,   5,      1702,    1500,    1500,    1500,    100,   1000,  True,       0,     True])
game_config_sets.append([748,   5,      1802,    1700,    1700,    1700,    100,   1000,  False,      0,     True])
game_config_sets.append([749,   2,      1902,    1800,    1800,    1800,    100,   1000,  False,      0,     True])
game_config_sets.append([750,   2,      2002,    0,       0,       0,       100,   1000,  True,       0,     True])
game_config_sets.append([751,   2,      2102,    100,     100,     100,     100,   1000,  True,       0,     True])
game_config_sets.append([752,   3,      2202,    400,     400,     400,     100,   1000,  True,       0,     True])
game_config_sets.append([753,   3,      2302,    600,     600,     600,     100,   1000,  False,      0,     True])
game_config_sets.append([754,   3,      2402,    900,     900,     900,     100,   1000,  False,      0,     True])
game_config_sets.append([755,   4,      2502,    1000,    1000,    1000,    100,   1000,  True,       0,     True])
game_config_sets.append([756,   4,      2602,    1200,    1200,    1200,    100,   1000,  False,      0,     True])
game_config_sets.append([757,   5,      2702,    1500,    1500,    1500,    100,   1000,  True,       0,     True])
game_config_sets.append([758,   5,      2802,    1700,    1700,    1700,    100,   1000,  False,      0,     True])
game_config_sets.append([759,   2,      2902,    1800,    1800,    1800,    100,   1000,  False,      0,     True])
game_config_sets.append([760,   2,      3002,    0,       0,       0,       100,   1000,  True,       0,     True])
game_config_sets.append([761,   2,      3102,    100,     100,     100,     100,   1000,  True,       0,     True])
game_config_sets.append([762,   3,      3202,    400,     400,     400,     100,   1000,  True,       0,     True])
game_config_sets.append([763,   3,      3302,    600,     600,     600,     100,   1000,  False,      0,     True])
game_config_sets.append([764,   3,      3402,    900,     900,     900,     100,   1000,  False,      0,     True])
game_config_sets.append([765,   4,      3502,    1000,    1000,    1000,    100,   1000,  True,       0,     True])
game_config_sets.append([766,   4,      3602,    1200,    1200,    1200,    100,   1000,  False,      0,     True])
game_config_sets.append([767,   5,      3702,    1500,    1500,    1500,    100,   1000,  True,       0,     True])
game_config_sets.append([768,   5,      3802,    1700,    1700,    1700,    100,   1000,  False,      0,     True])
game_config_sets.append([769,   2,      3902,    1800,    1800,    1800,    100,   1000,  False,      0,     True])



########################################
# id 800 - 899 are offical UT test config
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games, keep_batch, demos, flag_4in1
game_config_sets.append([800,   1,      100,     100,     100,     100,     10,    10,    True,       10,    True])  #temp: games/demos = 0->10
game_config_sets.append([801,   3,      100,     100,     100,     100,     10,    10,    True,       10,    True])
game_config_sets.append([802,   5,      100,     100,     100,     100,     10,    10,    True,       10,    True])

#as sanity verify: all kinds of agent with 1by1. regardless env(2,3)
game_config_sets.append([810,   3,      0,       100,     200,     300,     3,     10,    True,       1,     False])
game_config_sets.append([811,   2,      400,     500,     600,     700,     3,     10,    True,       1,     False])
game_config_sets.append([812,   4,      800,     900,     900,     900,     3,     10,    False,      1,     False])
game_config_sets.append([813,   5,      1000,    1100,    1200,    1300,    3,     10,    False,      1,     False])
game_config_sets.append([814,   1,      1400,    1500,    1600,    1700,    3,     10,    False,      1,     False])
game_config_sets.append([815,   0,      1800,    1900,    2000,    2100,    3,     10,    True,       1,     False])
game_config_sets.append([816,   2,      2300,    2400,    2600,    2700,    3,     10,    False,      1,     False])
game_config_sets.append([817,   2,      2800,    2900,    3000,    3100,    3,     10,    False,      1,     False])
game_config_sets.append([818,   2,      3200,    3300,    3400,    3500,    3,     10,    False,      1,     False])
game_config_sets.append([819,   2,      3600,    3700,    3800,    3900,    3,     10,    False,      1,     False])

#as sanity verify: all kinds of agent with 4in1. regardless env(2,3)
game_config_sets.append([820,   2,      0,       0,       0,       0,       3,     10,    True,       1,     True])
game_config_sets.append([821,   2,      100,     100,     100,     100,     3,     10,    True,       1,     True])
game_config_sets.append([822,   3,      500,     400,     400,     400,     3,     10,    True,       1,     True])
game_config_sets.append([823,   3,      600,     600,     600,     600,     3,     10,    False,      1,     True])
game_config_sets.append([824,   3,      800,     900,     900,     900,     3,     10,    False,      1,     True])
game_config_sets.append([825,   4,      1000,    1000,    1000,    1000,    3,     10,    True,       1,     True])
game_config_sets.append([826,   4,      1200,    1200,    1200,    1200,    3,     10,    False,      1,     True])
game_config_sets.append([827,   5,      1500,    1500,    1500,    1500,    3,     10,    True,       1,     True])
game_config_sets.append([828,   5,      1700,    1700,    1700,    1700,    3,     10,    False,      1,     True])
game_config_sets.append([829,   2,      1800,    1800,    1800,    1800,    3,     10,    False,      1,     True])
game_config_sets.append([830,   2,      2000,    2000,    2000,    2000,    3,     10,    False,      1,     True])
game_config_sets.append([831,   2,      2200,    2200,    2200,    2200,    3,     10,    False,      1,     True])
game_config_sets.append([832,   2,      2400,    2400,    2400,    2400,    3,     10,    False,      1,     True])
game_config_sets.append([833,   2,      2600,    2600,    2600,    2600,    3,     10,    False,      1,     True])
game_config_sets.append([834,   2,      2700,    2700,    2700,    2700,    3,     10,    False,      1,     True])
game_config_sets.append([835,   2,      2800,    2800,    2800,    2800,    3,     10,    False,      1,     True])
game_config_sets.append([836,   2,      3000,    3000,    3000,    3000,    3,     10,    False,      1,     True])
game_config_sets.append([837,   2,      3100,    3100,    3100,    3100,    3,     10,    False,      1,     True])
game_config_sets.append([838,   2,      3200,    3200,    3200,    3200,    3,     10,    False,      1,     True])
game_config_sets.append([839,   2,      3300,    3300,    3300,    3300,    3,     10,    False,      1,     True])
game_config_sets.append([840,   2,      3500,    3500,    3500,    3500,    3,     10,    False,      1,     True])
game_config_sets.append([841,   2,      3600,    3600,    3600,    3600,    3,     10,    False,      1,     True])
game_config_sets.append([842,   2,      3800,    3800,    3800,    3800,    3,     10,    False,      1,     True])


########################################
# id 900 - 999 are offical CPU measurement cases
# fastest: true-true-4in1
# slowest: false-false-not4in1
# agent 100, 300, 400, 317, 311 are strongest agents in round 1 competition(v4.2)
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games, keep_batch, demos, flag_4in1
game_config_sets.append([900,   1,      100,     100,     100,     100,     50,    5000,  True,       100,  True])
game_config_sets.append([901,   1,      100,     100,     100,     100,     100,   5000,  True,       100,  True])
game_config_sets.append([902,   1,      300,     400,     317,     311,     50,    5000,  True,       100,  False])

game_config_sets.append([903,   0,      300,     400,     317,     311,     50,    5000,  False,      100,  False])
game_config_sets.append([904,   0,      300,     400,     317,     311,     100,   5000,  False,      100,  False])
game_config_sets.append([905,   0,      100,     100,     100,     100,     50,    5000,  False,      100,  True])


##############################
# env=2
# false-false for general training
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1000,  2,      1,       4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1001,  2,      2,       10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1002,  2,      3,       104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1003,  2,      4,       4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1004,  2,      5,       10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1005,  2,      6,       104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1006,  2,      7,       4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1007,  2,      8,       10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1008,  2,      9,       104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1009,  2,      10,      4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1010,  2,      11,      10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1011,  2,      12,      104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1012,  2,      101,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1013,  2,      102,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1014,  2,      103,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1015,  2,      104,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1016,  2,      105,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1017,  2,      106,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1018,  2,      107,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1019,  2,      108,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1020,  2,      109,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1021,  2,      110,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1022,  2,      111,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1023,  2,      112,     104,     104,     104,     100,   500000,  False,      0,     True])
                                                                                                        
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1024,  2,      201,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1025,  2,      202,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1026,  2,      203,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1027,  2,      204,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1028,  2,      205,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1029,  2,      206,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1030,  2,      207,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1031,  2,      208,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1032,  2,      209,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1033,  2,      210,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1034,  2,      211,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1035,  2,      212,     4,       4,       4,       100,   500000,  False,      0,     True])
                                                                                                        
                                                                                                        
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1036,  2,      301,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1037,  2,      302,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1038,  2,      303,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1039,  2,      304,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1040,  2,      305,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1041,  2,      306,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1042,  2,      307,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1043,  2,      308,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1044,  2,      309,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1045,  2,      310,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1046,  2,      311,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1047,  2,      312,     4,       4,       4,       100,   500000,  False,      0,     True])
                         
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1048,  2,      401,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1049,  2,      402,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1050,  2,      403,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1051,  2,      404,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1052,  2,      405,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1053,  2,      406,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1054,  2,      407,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1055,  2,      408,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1056,  2,      409,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1057,  2,      410,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1058,  2,      411,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1059,  2,      412,     104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1060,  2,      501,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1061,  2,      502,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1062,  2,      503,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1063,  2,      504,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1064,  2,      505,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1065,  2,      506,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1066,  2,      507,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1067,  2,      508,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1068,  2,      509,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1069,  2,      510,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1070,  2,      511,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1071,  2,      512,     104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1072,  2,      601,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1073,  2,      602,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1074,  2,      603,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1075,  2,      604,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1076,  2,      605,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1077,  2,      606,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1078,  2,      607,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1079,  2,      608,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1080,  2,      609,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1081,  2,      610,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1082,  2,      611,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1083,  2,      612,     104,     104,     104,     100,   500000,  False,      0,     True])



#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1084,  2,      701,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1085,  2,      702,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1086,  2,      703,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1087,  2,      704,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1088,  2,      705,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1089,  2,      706,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1090,  2,      707,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1091,  2,      708,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1092,  2,      709,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1093,  2,      710,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1094,  2,      711,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1095,  2,      712,     104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1096,  2,      801,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1097,  2,      802,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1098,  2,      803,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1099,  2,      804,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1100,  2,      805,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1101,  2,      806,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1102,  2,      807,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1103,  2,      808,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1104,  2,      809,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1105,  2,      810,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1106,  2,      811,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1107,  2,      812,     104,     104,     104,     100,   500000,  False,      0,     True])
                                                                                                       

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1108,  2,      901,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1109,  2,      902,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1110,  2,      903,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1111,  2,      904,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1112,  2,      905,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1113,  2,      906,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1114,  2,      907,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1115,  2,      908,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1116,  2,      909,     4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1117,  2,      910,     10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1118,  2,      911,     104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1119,  2,      912,     104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1120,  2,      1001,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1121,  2,      1002,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1122,  2,      1003,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1123,  2,      1004,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1124,  2,      1005,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1125,  2,      1006,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1126,  2,      1007,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1127,  2,      1008,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1128,  2,      1009,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1129,  2,      1010,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1130,  2,      1011,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1131,  2,      1012,    104,     104,     104,     100,   500000,  False,      0,     True])



#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1132,  2,      1101,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1133,  2,      1102,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1134,  2,      1103,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1135,  2,      1104,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1136,  2,      1105,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1137,  2,      1106,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1138,  2,      1107,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1139,  2,      1108,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1140,  2,      1109,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1141,  2,      1110,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1142,  2,      1111,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1143,  2,      1112,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1144,  2,      1201,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1145,  2,      1202,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1146,  2,      1203,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1147,  2,      1204,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1148,  2,      1205,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1149,  2,      1206,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1150,  2,      1207,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1151,  2,      1208,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1152,  2,      1209,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1153,  2,      1210,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1154,  2,      1211,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1155,  2,      1212,    104,     104,     104,     100,   500000,  False,      0,     True])



#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1156,  2,      1301,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1157,  2,      1302,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1158,  2,      1303,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1159,  2,      1304,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1160,  2,      1305,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1161,  2,      1306,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1162,  2,      1307,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1163,  2,      1308,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1164,  2,      1309,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1165,  2,      1310,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1166,  2,      1311,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1167,  2,      1312,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1168,  2,      1401,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1169,  2,      1402,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1170,  2,      1403,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1171,  2,      1404,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1172,  2,      1405,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1173,  2,      1406,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1174,  2,      1407,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1175,  2,      1408,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1176,  2,      1409,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1177,  2,      1410,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1178,  2,      1411,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1179,  2,      1412,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1180,  2,      1501,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1181,  2,      1502,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1182,  2,      1503,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1183,  2,      1504,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1184,  2,      1505,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1185,  2,      1506,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1186,  2,      1507,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1187,  2,      1508,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1188,  2,      1509,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1189,  2,      1510,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1190,  2,      1511,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1191,  2,      1512,    104,     104,     104,     100,   500000,  False,      0,     True])



#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1192,  2,      1601,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1193,  2,      1602,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1194,  2,      1603,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1195,  2,      1604,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1196,  2,      1605,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1197,  2,      1606,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1198,  2,      1607,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1199,  2,      1608,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1200,  2,      1609,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1201,  2,      1610,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1202,  2,      1611,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1203,  2,      1612,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1204,  2,      1701,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1205,  2,      1702,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1206,  2,      1703,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1207,  2,      1704,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1208,  2,      1705,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1209,  2,      1706,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1210,  2,      1707,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1211,  2,      1708,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1212,  2,      1709,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1213,  2,      1710,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1214,  2,      1711,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1215,  2,      1712,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1216,  2,      1801,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1217,  2,      1802,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1218,  2,      1803,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1219,  2,      1804,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1220,  2,      1805,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1221,  2,      1806,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1222,  2,      1807,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1223,  2,      1808,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1224,  2,      1809,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1225,  2,      1810,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1226,  2,      1811,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1227,  2,      1812,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1228,  2,      1901,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1229,  2,      1902,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1230,  2,      1903,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1231,  2,      1904,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1232,  2,      1905,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1233,  2,      1906,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1234,  2,      1907,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1235,  2,      1908,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1236,  2,      1909,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1237,  2,      1910,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1238,  2,      1911,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1239,  2,      1912,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1240,  2,      2001,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1241,  2,      2002,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1242,  2,      2003,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1243,  2,      2004,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1244,  2,      2005,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1245,  2,      2006,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1246,  2,      2007,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1247,  2,      2008,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1248,  2,      2009,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1249,  2,      2010,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1250,  2,      2011,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1251,  2,      2012,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1252,  2,      2101,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1253,  2,      2102,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1254,  2,      2103,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1255,  2,      2104,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1256,  2,      2105,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1257,  2,      2106,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1258,  2,      2107,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1259,  2,      2108,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1260,  2,      2109,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1261,  2,      2110,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1262,  2,      2111,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1263,  2,      2112,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1264,  2,      2201,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1265,  2,      2202,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1266,  2,      2203,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1267,  2,      2204,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1268,  2,      2205,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1269,  2,      2206,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1270,  2,      2207,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1271,  2,      2208,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1272,  2,      2209,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1273,  2,      2210,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1274,  2,      2211,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1275,  2,      2212,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1276,  2,      2301,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1277,  2,      2302,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1278,  2,      2303,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1279,  2,      2304,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1280,  2,      2305,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1281,  2,      2306,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1282,  2,      2307,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1283,  2,      2308,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1284,  2,      2309,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1285,  2,      2310,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1286,  2,      2311,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1287,  2,      2312,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1288,  2,      2401,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1289,  2,      2402,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1290,  2,      2403,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1291,  2,      2404,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1292,  2,      2405,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1293,  2,      2406,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1294,  2,      2407,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1295,  2,      2408,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1296,  2,      2409,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1297,  2,      2410,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1298,  2,      2411,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1299,  2,      2412,    104,     104,     104,     100,   500000,  False,      0,     True])



#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1300,  2,      2501,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1301,  2,      2502,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1302,  2,      2503,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1303,  2,      2504,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1304,  2,      2505,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1305,  2,      2506,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1306,  2,      2507,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1307,  2,      2508,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1308,  2,      2509,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1309,  2,      2510,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1310,  2,      2511,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1311,  2,      2512,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1312,  2,      2601,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1313,  2,      2602,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1314,  2,      2603,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1315,  2,      2604,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1316,  2,      2605,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1317,  2,      2606,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1318,  2,      2607,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1319,  2,      2608,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1320,  2,      2609,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1321,  2,      2610,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1322,  2,      2611,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1323,  2,      2612,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1324,  2,      2701,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1325,  2,      2702,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1326,  2,      2703,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1327,  2,      2704,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1328,  2,      2705,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1329,  2,      2706,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1330,  2,      2707,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1331,  2,      2708,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1332,  2,      2709,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1333,  2,      2710,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1334,  2,      2711,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1335,  2,      2712,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1336,  2,      2801,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1337,  2,      2802,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1338,  2,      2803,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1339,  2,      2804,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1340,  2,      2805,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1341,  2,      2806,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1342,  2,      2807,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1343,  2,      2808,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1344,  2,      2809,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1345,  2,      2810,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1346,  2,      2811,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1347,  2,      2812,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1348,  2,      2901,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1349,  2,      2902,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1350,  2,      2903,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1351,  2,      2904,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1352,  2,      2905,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1353,  2,      2906,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1354,  2,      2907,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1355,  2,      2908,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1356,  2,      2909,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1357,  2,      2910,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1358,  2,      2911,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1359,  2,      2912,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1360,  2,      3001,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1361,  2,      3002,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1362,  2,      3003,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1363,  2,      3004,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1364,  2,      3005,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1365,  2,      3006,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1366,  2,      3007,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1367,  2,      3008,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1368,  2,      3009,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1369,  2,      3010,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1370,  2,      3011,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1371,  2,      3012,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1372,  2,      3101,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1373,  2,      3102,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1374,  2,      3103,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1375,  2,      3104,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1376,  2,      3105,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1377,  2,      3106,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1378,  2,      3107,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1379,  2,      3108,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1380,  2,      3109,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1381,  2,      3110,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1382,  2,      3111,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1383,  2,      3112,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1384,  2,      3201,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1385,  2,      3202,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1386,  2,      3203,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1387,  2,      3204,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1388,  2,      3205,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1389,  2,      3216,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1390,  2,      3217,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1391,  2,      3218,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1392,  2,      3219,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1393,  2,      3210,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1394,  2,      3211,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1395,  2,      3212,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1396,  2,      3301,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1397,  2,      3302,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1398,  2,      3303,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1399,  2,      3304,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1400,  2,      3305,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1401,  2,      3306,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1402,  2,      3307,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1403,  2,      3308,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1404,  2,      3309,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1405,  2,      3310,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1406,  2,      3311,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1407,  2,      3312,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1408,  2,      3401,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1409,  2,      3402,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1410,  2,      3403,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1411,  2,      3404,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1412,  2,      3405,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1413,  2,      3406,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1414,  2,      3407,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1415,  2,      3408,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1416,  2,      3409,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1417,  2,      3410,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1418,  2,      3411,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1419,  2,      3412,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1420,  2,      3501,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1421,  2,      3502,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1422,  2,      3503,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1423,  2,      3504,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1424,  2,      3505,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1425,  2,      3506,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1426,  2,      3507,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1427,  2,      3508,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1428,  2,      3509,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1429,  2,      3510,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1430,  2,      3511,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1431,  2,      3512,    104,     104,     104,     100,   500000,  False,      0,     True])

#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1432,  2,      3601,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1433,  2,      3602,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1434,  2,      3603,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1435,  2,      3604,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1436,  2,      3605,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1437,  2,      3606,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1438,  2,      3607,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1439,  2,      3608,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1440,  2,      3609,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1441,  2,      3610,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1442,  2,      3611,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1443,  2,      3612,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1444,  2,      3701,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1445,  2,      3702,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1446,  2,      3703,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1447,  2,      3704,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1448,  2,      3705,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1449,  2,      3706,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1450,  2,      3707,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1451,  2,      3708,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1452,  2,      3709,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1453,  2,      3710,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1454,  2,      3711,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1455,  2,      3712,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1456,  2,      3801,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1457,  2,      3802,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1458,  2,      3803,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1459,  2,      3804,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1460,  2,      3805,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1461,  2,      3806,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1462,  2,      3807,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1463,  2,      3808,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1464,  2,      3809,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1465,  2,      3810,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1466,  2,      3811,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1467,  2,      3812,    104,     104,     104,     100,   500000,  False,      0,     True])


#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([1468,  2,      3901,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1469,  2,      3902,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1470,  2,      3903,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1471,  2,      3904,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1472,  2,      3905,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1473,  2,      3906,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1474,  2,      3907,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1475,  2,      3908,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1476,  2,      3909,    4,       4,       4,       100,   500000,  False,      0,     True])
game_config_sets.append([1477,  2,      3910,    10,      10,      10,      100,   500000,  False,      0,     True])
game_config_sets.append([1478,  2,      3911,    104,     104,     104,     100,   500000,  False,      0,     True])
game_config_sets.append([1479,  2,      3912,    104,     104,     104,     100,   500000,  False,      0,     True])




##############################
# env=4
# false-false for general training
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1




def generate_competition_game_config(game_id_start, all_demos, batch, pattern1):

    import os
    import re
    from itertools import combinations
    
    file_name_list = os.listdir("./results/")

    #pattern = 'play_e_0/1/2/3/4/5', 只有pattern一样的.h5 file才能匹配
    #1.search pattern
    #2.remove pattern and '.h5'
    #split by '_', keep first part -> should be result
    #remove 10的整倍数，那些是debug purpose
    pattern1 = 'play_e_2_'
    pattern2 = '.h5'
    pattern3 = r'_g.'   #_gd, _gc
    is_play_e_h5 = [ name for name in file_name_list if name.find(pattern1, 0) >= 0 and name.find(pattern2, 0) >=0 ]
    is_agent_gx = [ re.split(r'play_e_2_|.h5', name)[1] for name in is_play_e_h5 ]
    is_agent = [ int(re.sub(pattern3, '', name)) for name in is_agent_gx ]
    agent_list = [ agent_id for agent_id in is_agent if agent_id % 10 !=0 ]
    agent_set = set(agent_list)
    
    for i in combinations(agent_set, 2):
        print(i)
        
        
        
        
    total_agents = len(agent_set)
    print("total agents ", total_agents, all_competitions)
    count = 0                   
    f = open('./results/generated_games.txt', 'w+')
    game_lines = []             
                                
    for env_id in [0, 2]:  # 2 kinds of env. reward templates. competition(=demo) regardless of 'reward_template'
        for i in range(0, total_agents-1, 1):
            sn_agent = agent_config_sets[i][0] #id_agent_id=0
            for j in range(i+1, total_agents, 1):
                ew_agent = agent_config_sets[j][0]
                
                game_id = game_id_start + count
                #env_id = 0 #only 0
                agent_class_s = sn_agent
                agent_class_e = ew_agent
                agent_class_n = sn_agent
                agent_class_w = ew_agent
                batch_size = batch
                games = 0  #no training
                keep_batch_same_in_train = False
                demos = all_demos
                for_in_one = False
                #game_config_sets.append([10000, 0, 0, 2, 0, 2, 100,50, False, 10000, False])            
                game_line = 'game_config_sets.append([' + str(game_id) + ',\t' + str(env_id) +',\t' + str(agent_class_s) + ',\t\t' +        \
                            str(agent_class_e) + ',\t\t' + str(agent_class_n) + ',\t\t' + str(agent_class_w) + ',\t\t' + str(batch_size) +  \
                            ',\t' + str(games) + ',\t' + 'False' + ',\t' + str(demos) + ',\t' + 'False' + '])' + '\n'
                f.write(game_line)
                game_lines.append(game_line)
                
                count += 1
                if count >= all_competitions:
                    break
            if count >= all_competitions:
                break
        
    f.close()

        
game_id_start = 100100  #officail competition from ID 100100
end_ids = [2]
all_games = len(agent_config_sets) * (len(agent_config_sets)-1) / 2  #8010/2
all_demos = 10000
batch = 50
#generate_competition_game_config(game_id_start, all_games, all_demos, batch)

    
########################################
# id: [100000, 100099] test competition
#competetion: ToDo : rename the gameid in existing .h5 file: gameid-envid-agentid ==> new gameid(for competetion)-envid-agentid

#BELOW IS AUOTO GENERATED GAMES for COMPETITION
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([100000,	2,	3600,		3603,		3600,		3603,		50,	0,	False,	200,	False])
game_config_sets.append([100001,	0,	1,		2,		1,		2,		50,	0,	False,	200,	False])
game_config_sets.append([100002,	0,	3,		5,		3,		5,		50,	0,	False,	200,	False])
game_config_sets.append([100003,	0,	4,		7,		4,		7,		50,	0,	False,	200,	False])
game_config_sets.append([100004,	0,	6,		2,		6,		2,		50,	0,	False,	200,	False])
game_config_sets.append([100005,	0,	1,		3,		1,		3,		50,	0,	False,	200,	False])
game_config_sets.append([100006,	0,	6,		4,		6,		4,		50,	0,	False,	200,	False])

########################################
# id >= 100100 are offical competition
#competetion: ToDo : rename the gameid in existing .h5 file: gameid-envid-agentid ==> new gameid(for competetion)-envid-agentid
#BELOW IS AUOTO GENERATED GAMES for COMPETITION
#                        id,    env_id, agent_s, agent_e, agent_n, agent_w, batch, games,   keep_batch, demos, flag_4in1
game_config_sets.append([100100,	0,	0,		1,		0,		1,		50,	0,	False,	10000,	False])
game_config_sets.append([100101,	0,	0,		2,		0,		2,		50,	0,	False,	10000,	False])
game_config_sets.append([100102,	0,	0,		3,		0,		3,		50,	0,	False,	10000,	False])
game_config_sets.append([100103,	0,	0,		4,		0,		4,		50,	0,	False,	10000,	False])
game_config_sets.append([100104,	0,	0,		5,		0,		5,		50,	0,	False,	10000,	False])
game_config_sets.append([100105,	0,	0,		6,		0,		6,		50,	0,	False,	10000,	False])
game_config_sets.append([100106,	0,	0,		7,		0,		7,		50,	0,	False,	10000,	False])
game_config_sets.append([100107,	0,	0,		8,		0,		8,		50,	0,	False,	10000,	False])
game_config_sets.append([100108,	0,	0,		9,		0,		9,		50,	0,	False,	10000,	False])
game_config_sets.append([100109,	0,	0,		10,		0,		10,		50,	0,	False,	10000,	False])
game_config_sets.append([100110,	0,	0,		11,		0,		11,		50,	0,	False,	10000,	False])
game_config_sets.append([100111,	0,	0,		12,		0,		12,		50,	0,	False,	10000,	False])
game_config_sets.append([100112,	0,	0,		13,		0,		13,		50,	0,	False,	10000,	False])
game_config_sets.append([100113,	0,	0,		14,		0,		14,		50,	0,	False,	10000,	False])
game_config_sets.append([100114,	0,	0,		15,		0,		15,		50,	0,	False,	10000,	False])
game_config_sets.append([100115,	0,	0,		16,		0,		16,		50,	0,	False,	10000,	False])
game_config_sets.append([100116,	0,	0,		17,		0,		17,		50,	0,	False,	10000,	False])
game_config_sets.append([100117,	0,	0,		100,		0,		100,		50,	0,	False,	10000,	False])
game_config_sets.append([100118,	0,	0,		101,		0,		101,		50,	0,	False,	10000,	False])
game_config_sets.append([100119,	0,	0,		102,		0,		102,		50,	0,	False,	10000,	False])
game_config_sets.append([100120,	0,	0,		103,		0,		103,		50,	0,	False,	10000,	False])
game_config_sets.append([100121,	0,	0,		104,		0,		104,		50,	0,	False,	10000,	False])
game_config_sets.append([100122,	0,	0,		105,		0,		105,		50,	0,	False,	10000,	False])
game_config_sets.append([100123,	0,	0,		106,		0,		106,		50,	0,	False,	10000,	False])
game_config_sets.append([100124,	0,	0,		107,		0,		107,		50,	0,	False,	10000,	False])
game_config_sets.append([100125,	0,	0,		108,		0,		108,		50,	0,	False,	10000,	False])
game_config_sets.append([100126,	0,	0,		109,		0,		109,		50,	0,	False,	10000,	False])
game_config_sets.append([100127,	0,	0,		110,		0,		110,		50,	0,	False,	10000,	False])
game_config_sets.append([100128,	0,	0,		111,		0,		111,		50,	0,	False,	10000,	False])
game_config_sets.append([100129,	0,	0,		200,		0,		200,		50,	0,	False,	10000,	False])
game_config_sets.append([100130,	0,	0,		201,		0,		201,		50,	0,	False,	10000,	False])
game_config_sets.append([100131,	0,	0,		202,		0,		202,		50,	0,	False,	10000,	False])
game_config_sets.append([100132,	0,	0,		203,		0,		203,		50,	0,	False,	10000,	False])
game_config_sets.append([100133,	0,	0,		204,		0,		204,		50,	0,	False,	10000,	False])
game_config_sets.append([100134,	0,	0,		205,		0,		205,		50,	0,	False,	10000,	False])
game_config_sets.append([100135,	0,	0,		206,		0,		206,		50,	0,	False,	10000,	False])
game_config_sets.append([100136,	0,	0,		207,		0,		207,		50,	0,	False,	10000,	False])
game_config_sets.append([100137,	0,	0,		208,		0,		208,		50,	0,	False,	10000,	False])
game_config_sets.append([100138,	0,	0,		209,		0,		209,		50,	0,	False,	10000,	False])
game_config_sets.append([100139,	0,	0,		210,		0,		210,		50,	0,	False,	10000,	False])
game_config_sets.append([100140,	0,	0,		211,		0,		211,		50,	0,	False,	10000,	False])
game_config_sets.append([100141,	0,	0,		212,		0,		212,		50,	0,	False,	10000,	False])
game_config_sets.append([100142,	0,	0,		213,		0,		213,		50,	0,	False,	10000,	False])
game_config_sets.append([100143,	0,	0,		214,		0,		214,		50,	0,	False,	10000,	False])
game_config_sets.append([100144,	0,	0,		215,		0,		215,		50,	0,	False,	10000,	False])
game_config_sets.append([100145,	0,	0,		216,		0,		216,		50,	0,	False,	10000,	False])
game_config_sets.append([100146,	0,	0,		217,		0,		217,		50,	0,	False,	10000,	False])
game_config_sets.append([100147,	0,	0,		300,		0,		300,		50,	0,	False,	10000,	False])
game_config_sets.append([100148,	0,	0,		301,		0,		301,		50,	0,	False,	10000,	False])
game_config_sets.append([100149,	0,	0,		302,		0,		302,		50,	0,	False,	10000,	False])
game_config_sets.append([100150,	0,	0,		303,		0,		303,		50,	0,	False,	10000,	False])
game_config_sets.append([100151,	0,	0,		304,		0,		304,		50,	0,	False,	10000,	False])
game_config_sets.append([100152,	0,	0,		305,		0,		305,		50,	0,	False,	10000,	False])
game_config_sets.append([100153,	0,	0,		306,		0,		306,		50,	0,	False,	10000,	False])
game_config_sets.append([100154,	0,	0,		307,		0,		307,		50,	0,	False,	10000,	False])
game_config_sets.append([100155,	0,	0,		308,		0,		308,		50,	0,	False,	10000,	False])
game_config_sets.append([100156,	0,	0,		309,		0,		309,		50,	0,	False,	10000,	False])
game_config_sets.append([100157,	0,	0,		310,		0,		310,		50,	0,	False,	10000,	False])
game_config_sets.append([100158,	0,	0,		311,		0,		311,		50,	0,	False,	10000,	False])
game_config_sets.append([100159,	0,	0,		312,		0,		312,		50,	0,	False,	10000,	False])
game_config_sets.append([100160,	0,	0,		313,		0,		313,		50,	0,	False,	10000,	False])
game_config_sets.append([100161,	0,	0,		314,		0,		314,		50,	0,	False,	10000,	False])
game_config_sets.append([100162,	0,	0,		315,		0,		315,		50,	0,	False,	10000,	False])
game_config_sets.append([100163,	0,	0,		316,		0,		316,		50,	0,	False,	10000,	False])
game_config_sets.append([100164,	0,	0,		317,		0,		317,		50,	0,	False,	10000,	False])
game_config_sets.append([100165,	0,	0,		400,		0,		400,		50,	0,	False,	10000,	False])
game_config_sets.append([100166,	0,	0,		401,		0,		401,		50,	0,	False,	10000,	False])
game_config_sets.append([100167,	0,	0,		402,		0,		402,		50,	0,	False,	10000,	False])
game_config_sets.append([100168,	0,	0,		403,		0,		403,		50,	0,	False,	10000,	False])
game_config_sets.append([100169,	0,	0,		404,		0,		404,		50,	0,	False,	10000,	False])
game_config_sets.append([100170,	0,	0,		405,		0,		405,		50,	0,	False,	10000,	False])
game_config_sets.append([100171,	0,	0,		406,		0,		406,		50,	0,	False,	10000,	False])
game_config_sets.append([100172,	0,	0,		407,		0,		407,		50,	0,	False,	10000,	False])
game_config_sets.append([100173,	0,	0,		408,		0,		408,		50,	0,	False,	10000,	False])
game_config_sets.append([100174,	0,	0,		409,		0,		409,		50,	0,	False,	10000,	False])
game_config_sets.append([100175,	0,	0,		410,		0,		410,		50,	0,	False,	10000,	False])
game_config_sets.append([100176,	0,	0,		411,		0,		411,		50,	0,	False,	10000,	False])
game_config_sets.append([100177,	0,	0,		500,		0,		500,		50,	0,	False,	10000,	False])
game_config_sets.append([100178,	0,	0,		501,		0,		501,		50,	0,	False,	10000,	False])
game_config_sets.append([100179,	0,	0,		502,		0,		502,		50,	0,	False,	10000,	False])
game_config_sets.append([100180,	0,	0,		503,		0,		503,		50,	0,	False,	10000,	False])
game_config_sets.append([100181,	0,	0,		504,		0,		504,		50,	0,	False,	10000,	False])
game_config_sets.append([100182,	0,	0,		505,		0,		505,		50,	0,	False,	10000,	False])
game_config_sets.append([100183,	0,	0,		506,		0,		506,		50,	0,	False,	10000,	False])
game_config_sets.append([100184,	0,	0,		507,		0,		507,		50,	0,	False,	10000,	False])
game_config_sets.append([100185,	0,	0,		508,		0,		508,		50,	0,	False,	10000,	False])
game_config_sets.append([100186,	0,	0,		509,		0,		509,		50,	0,	False,	10000,	False])
game_config_sets.append([100187,	0,	0,		510,		0,		510,		50,	0,	False,	10000,	False])
game_config_sets.append([100188,	0,	0,		511,		0,		511,		50,	0,	False,	10000,	False])
game_config_sets.append([100189,	0,	1,		2,		1,		2,		50,	0,	False,	10000,	False])
game_config_sets.append([100190,	0,	1,		3,		1,		3,		50,	0,	False,	10000,	False])
game_config_sets.append([100191,	0,	1,		4,		1,		4,		50,	0,	False,	10000,	False])
game_config_sets.append([100192,	0,	1,		5,		1,		5,		50,	0,	False,	10000,	False])
game_config_sets.append([100193,	0,	1,		6,		1,		6,		50,	0,	False,	10000,	False])
game_config_sets.append([100194,	0,	1,		7,		1,		7,		50,	0,	False,	10000,	False])
game_config_sets.append([100195,	0,	1,		8,		1,		8,		50,	0,	False,	10000,	False])
game_config_sets.append([100196,	0,	1,		9,		1,		9,		50,	0,	False,	10000,	False])
game_config_sets.append([100197,	0,	1,		10,		1,		10,		50,	0,	False,	10000,	False])
game_config_sets.append([100198,	0,	1,		11,		1,		11,		50,	0,	False,	10000,	False])
game_config_sets.append([100199,	0,	1,		12,		1,		12,		50,	0,	False,	10000,	False])
game_config_sets.append([100200,	0,	1,		13,		1,		13,		50,	0,	False,	10000,	False])
