import numpy as np
import matplotlib.pyplot as plt

def main(PR, P1_8, data):
     
    def EEV(MI, prior):
        var_prior = np.var(prior)
        return (var_prior/(np.exp(2*MI)-1))

    EEV_P1 = EEV(data[0], P1_8[:,0])
    EEV_P2 = EEV(data[1], P1_8[:,1])
    EEV_P3 = EEV(data[2], P1_8[:,2])
    EEV_P4 = EEV(data[3], P1_8[:,3])
    EEV_P5 = EEV(data[4], P1_8[:,4])
    EEV_P6 = EEV(data[5], P1_8[:,5])
    EEV_P7 = EEV(data[6], P1_8[:,6])
    EEV_P8 = EEV(data[7], P1_8[:,7])

    out = np.vstack((EEV_P1, EEV_P2, EEV_P3, EEV_P4, EEV_P5, EEV_P6, EEV_P7, EEV_P8))
    print('Average EEV for PR'+str(PR)+' is: '+str(np.mean(out)))
    np.savetxt('./EE_results/PR'+str(PR)+'_EEV.txt', out)

if __name__ == '__main__':

    param_dist = np.loadtxt('./output/transformed/param_alpha_values.txt')
    
    for PR in range(1,8):
        MI_estimate = np.loadtxt('./EE_results/PR'+str(PR)+'_MI.txt')
        main(PR, param_dist, MI_estimate)
        continue

