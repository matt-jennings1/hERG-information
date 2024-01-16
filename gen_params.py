import numpy as np
import os
import multiprocessing
import prior

def main(n_desired=1000, tag=''):
    n_tot = 0

    dirname = 'output/'+str(n_desired)+str(tag)+'_params/'

    try:
        os.mkdir(dirname)
    except:
        print('Directory '+dirname+' already exists, proceeding...')
    else: 
        print('Successfully created output directory '+dirname+', proceeding...')

    while n_tot < n_desired: 

        params, numgen = prior.sample(int(1e7))

        for i in range(numgen):
            os.mkdir(dirname+'sample_'+str(n_tot)+'/')
            np.savetxt(dirname+'sample_'+str(n_tot)+'/'+'params_'+str(n_tot)+'.txt', params[:,i])
            n_tot += 1
            if n_tot == n_desired:
                break
            else:
                continue

    print('Saved a total of '+str(n_tot)+' parameter sets to: '+dirname)

if __name__ == '__main__':

    main(10000,'_Alpha')
    print("Task completed.")
