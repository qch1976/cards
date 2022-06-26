##################################################
# Generate competition result. LINUX only
# invoked by many_taskset2.py
#   if game_x_x is invoked by many_taskset, it performs single CPU run. 
#   the competition result can't be assembeled in game_x_x. 
#   then the only way is to generate the result from many_taskset
##################################################

import sys, getopt
import meas_5_2 as meas

#invoked by many_taskset2.py. to generate competition report. game_xx.py can't do it since it performs single run
def wrap_up_competition(selected_p_set, from_to):
    if from_to > 0:
        accum_meas = meas.Measurements(0) #0= dummy his
    
        #Here, 要求game config ID MUST连续！！！否则，-p i可能是不存在的. 执行"param_set.read_params(i)"才行
        accum_meas.game_ids = list(range(selected_p_set, selected_p_set + from_to,1))  #Here, 要求game config ID MUST连续！！！
    
        accum_meas.assemble_records()  #collect records.csv created by demo()
        accum_meas.analyze_competition_result()

        print("wrap_comp: created competition report")
    else:
        print("wrap_comp: wrong -t = 0")


#python wrap_comp.py -t 6 -p 100000
def main(argv):
    print("argv ", argv)

    selected_p_set = []
    from_to = 0

    ####################
    # get command line input params
    ####################
    try:
        opts, args = getopt.getopt(argv,"t:p:")
    except getopt.GetoptError:
        print("wrong input")
        return;

    try:
        for opt, arg in opts:
            print("arg: ",opt, arg)
            if opt == '-t':
                from_to = int(arg)

            if opt == '-p':
                selected_p_set0 = arg.split(',')
                selected_p_set1 = [int(c) for c in selected_p_set0] #support only 1 param set
                selected_p_set = selected_p_set1[0]
                
    except  ValueError:
        print("wrong input", opt, arg)
        return

    print("input set: param set id + all games: ", selected_p_set, from_to)
    wrap_up_competition(selected_p_set, from_to)

    return


if __name__ == "__main__":
    # temp test field

    # temp test field
    
    ######################################
    # offical starting ...
    ######################################
    main(sys.argv[1:])
