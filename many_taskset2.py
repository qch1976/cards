###########################################
# generate : LINUX only
# taskset -c xx python game_5_2.py -r -init -p 1000
# etc...
# affinity can't take 100% CPU. but taskset can.

# v0.1: init on 5.2. Process(target...)
#  0.2: process pool
##################################################

import os
import numpy as np
import sys, getopt
from multiprocessing import Process, Pool, set_start_method



def run_child(taskset_cmd_line):
    print("run child: ", taskset_cmd_line)
    os.system(taskset_cmd_line)  #it will run itself again and again



def main(argv):
    print("argv ", argv)
#python many_taskset2.py -r init -m 6 -t 18 -p 4 -d 2 -f discard_testbench_5_2.py > d2

# command line is same to 'game_xxx.py"
#-r init -m 3 -c 7 -s 13 -p 1
#-r init -m 3 -c 7 -g 0 -s 13 -p 1   #-g=*10 in loop in discard only
#-r init -p 4
#-r init -m 3 -c 7 -s 13 -t 50 -p 1000        #from p1000, run 50 configs continuous, on CPU 5,6,7
#-r init -m 3 -c 7 -s 13 -t 50 -p 1000 -d 1   #d=1, cmd_line1=taskset
#-r init -m 3 -c 7 -s 13 -t 50 -p 1000 -d 2   #d=2, cmd_line2=no CPU specified
#-r init -m 3 -c 7 -s 13 -t 50 -p 1000 -d 2 -f discard_testbench_5_2.py  #with main_file
#-r resume -m 3 -c 7 -s 13 -p 0
#-r resume -m 3 -c 7 -s 13 -t 50 -p 0     #from p1000, run 50 configs, on CPU 5,6,7
#NOTE: if need not strict CPU assignment, don't use many_taskset.py. python CPU affinity in game_xx.py would propagate CPU utilization to more CPUs

    multi_proces = 0
    cpu_back_start = 7
    seed_start = 13
    selected_p_set = []
    from_to = 0
    cmdid = 1
    gpu_id = 0
    enable_GPU = False
    main_file = "game_5_2.py"  #as default #discard_testbench_5_2.py
    wrap_comp_file = "wrap_comp.py"
    
    #spawn: start from 'target'. fork: start from 'now on'
    set_start_method('spawn', force=True) #fork does not work, multi-processes can't be started

    ####################
    # get command line input params
    ####################
    try:
        opts, args = getopt.getopt(argv,"r:m:c:s:p:a:u:t:d:f:g:")
    except getopt.GetoptError:
        print("wrong input")
        return;

    try:
        for opt, arg in opts:
            print("arg: ",opt, arg)
            if opt == '-r':
                if arg == 'init' :
                    sub_cmd = 'init'
                elif arg == 'resume' :
                    sub_cmd = 'resume'
                elif arg == 'comp' :
                    sub_cmd = 'comp'
                else:
                    print("wrong -r input", opt, arg)
                    return

            if opt == '-m': #multi process: 0,1-27
                multi_proces = int(arg)

            if opt == '-c': #7 or 27
                cpu_back_start = int(arg)

            if opt == '-s':
                seed_start = int(arg)
            
            if opt == '-t':
                from_to = int(arg)

            if opt == '-d':
                cmdid = int(arg)

            if opt == '-f':
                main_file = arg

            if opt == '-p':
                selected_p_set0 = arg.split(',')
                selected_p_set1 = [int(c) for c in selected_p_set0] #support only 1 param set
                selected_p_set = selected_p_set1[0]
                
            if opt == '-g':  #actually, -g doesn't work. enable/disable the GPU, have to do so before main during imported pkgs start up
                gpu_id = int(arg)
                enable_GPU = True

    except  ValueError:
        print("wrong input", opt, arg)
        return

    print("input set: multi-porcess + cpu start + seed start + param set id + all games: ", multi_proces, cpu_back_start, seed_start, selected_p_set, from_to)

        
    ##############################
    # multiple processes startup
    ##############################
    if from_to > 0 and multi_proces > 0:
        #init or resume with 'from ... to ...', training() only
        print("YDL: bundle training starting ", selected_p_set, from_to)
        po = Pool(multi_proces)  # 最大的进程数
        
        for i in range(selected_p_set, selected_p_set + from_to,1):  #Here, 要求game config ID MUST连续！！！否则，-p i可能是不存在的. 执行"param_set.read_params(i)"才行
            #注意 selected_p_set和selected_p_set2的区别
            print("YDL: config line ", i, multi_proces)
            #seed_offset = 0 #for single deal set training only
            seed_offset = int(np.random.random_sample() * 1000000) % 65535  #for general training with various seed
            
            cpu_offset = (i - selected_p_set) % multi_proces
            cpu = cpu_back_start-cpu_offset
            seed = seed_start+seed_offset
        
            cmd_line1 = "taskset -c " + str(cpu) + " python " + main_file + " -r " + sub_cmd + " -s " + str(seed) + " -p " + str(i)
            cmd_line2 = "python " + main_file + " -r " + sub_cmd + " -s " + str(seed) + " -p " + str(i)
            if True == enable_GPU:
                cmd_line1 += ' -g ' + str(gpu_id)
                cmd_line2 += ' -g ' + str(gpu_id)
            
            if 1 == cmdid:  #linux only
                cmd_line = cmd_line1  #linux only
            elif 2 == cmdid:
                cmd_line = cmd_line2

            print(cmd_line)
            #p = Process(target=run_child, args=([cmd_line]))
            po.apply_async(run_child,(cmd_line,))
                
            
        print("YDL: ", from_to, " processes start in pools ", multi_proces)
        po.close()    # 关闭进程池，关闭后po不再接受新的请求
        po.join()     # 等待po中的所有子进程执行完成，必须放在close语句之后
        
    ##############################
    # generate competition report
    ##############################
    if 'comp' == sub_cmd and from_to > 0:   #if only one competition, doesn't create report
        cmd_line = " python " + wrap_comp_file + " -t " + str(from_to) + " -p " + str(selected_p_set)
        print(cmd_line)
        po.apply_async(run_child,(cmd_line,))

    return


if __name__ == "__main__":
    # temp test field

    # temp test field
    
    ######################################
    # offical starting ...
    ######################################
    main(sys.argv[1:])
