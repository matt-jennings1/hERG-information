import numpy as np
import os
import subprocess
from pr_lengths import protocol_length as prl

def main(PR=7, n=1000,tag=''):

#Desired length of solution-- curves shorter than this indicate CVODE failed to converge
    L = prl(PR)

#protocol.txt is read by the executable and determines the simulated protocol.
    f = open('protocol.txt', 'w')
    f.write(str(PR))
    f.close()

#Copies the parameter sets into local directory and solves them. Results moved back to
#appropriate sample directory.
    for x in range(n):
        os.system('cp output/'+str(n)+str(tag)+'_params/sample_'+str(x)+'/params_'+str(x)+'.txt input.txt')
        subprocess.run('./HHFullOut')
        temp = np.loadtxt('hh.out') 
        
        if len(temp) != L:
            print('Model failed to converge for sample #'+str(x))
            continue
        else:
            np.save('./output/'+str(n)+str(tag)+'_params/sample_'+str(x)+'/PR'+str(PR)+'_output_'+str(x)+'.npy', temp[:,5])
            print('Completed sample number: '+str(x))

    print('Successfully ran '+str(n)+' sets of parameters.')

if __name__ == '__main__':
    
    for x in range(1,8):
        main(PR=x, n=10000, tag='_Alpha')
