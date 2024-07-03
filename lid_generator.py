import pickle
import subprocess
import time
import argparse
import lid_vis

if __name__=="__main__":

    #Build in argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--flags', type=str, default='_',
                        help = 'Any (optional) flag which needs ' +
                        'to be passed to caller can be done so here')
    args, _ = parser.parse_known_args()
    
    #Parallel stuff
    MaxProcesses = 1
    Processes = []

    def checkrunning():
        for p in reversed(range(len(Processes))):
            if Processes[p].poll() is not None:
                del Processes[p]
        return len(Processes)
            
    #Fix some parameters
    N = 20
    dt = 0.001
    Mbase = 10

    #Define lists for those we want to vary
    Mrefs = [2,3,4]
    tdegrees = [0,1]
    sdegrees = [1,2]
    Rs = [1e0,1e1,1e2]
    MPIProcesses = 8

    #Make temporary file name unique under flags
    tmpname = args.flags.replace(' ', '_').replace('-','')
    
    #generate data
    for Mref in Mrefs:
        for tdegree in tdegrees:
            for sdegree in sdegrees:
                for R in Rs:
                    print('Mref =', Mref)
                    print('tdegree =', tdegree)
                    print('sdegree =', sdegree)
                    print('R =', R)

                    process = subprocess.Popen('mpiexec -n %s python lid_caller.py %s --N %s --dt %s --tdegree %s --sdegree %s --R %s --Mbase %s --Mref %s --tmpname %s' % (MPIProcesses, '--'+args.flags, N, dt, tdegree, sdegree, R, Mbase, Mref, tmpname),
                                               shell=True, stdout=subprocess.PIPE)
                    Processes.append(process)

                    while checkrunning()==MaxProcesses:
                        time.sleep(1)
    while checkrunning()!=0:
        time.sleep(1)

    print('Data generation complete')
    
    #Pick up data and in list
    data = []
    for Mref in Mrefs:
        for tdegree in tdegrees:
            for sdegree in sdegrees:
                for R in Rs:
                    filename = 'tmp/liddat%s%s%s%s%s%s%s%s%s%s' % (tmpname, MPIProcesses,
                                                                 N, dt, tdegree,
                                                                 sdegree, R, 1,
                                                                 Mbase, Mref)
                    try:
                        with open(filename, 'rb') as file:
                            out_ = pickle.load(file)
                            data_ = ['Mref %s, tdegree %s, sdegree %s, R %s' % (Mref,tdegree,sdegree,R),
                                     out_['dof'], out_['nnz'], out_['iterations'], out_['newton iterations'], out_['time_total'],out_['time_solve']]
                        data.append(data_)
                    except Exception as e:
                        print('Loading %s failed with' % filename)
                        print("Error : "+str(e))
                        
    
    print(data)
    #Write to file
    file = open('lid%s.pickle' % tmpname, 'wb')
    pickle.dump(data,file)
    file.close()

    #Try and visual
    try:
        lid_vis.visualise(tmpname)
    except Exception as e:
        print('Visualisation of %s failed with' % tmpname)
        print('Error : ', str(e))
