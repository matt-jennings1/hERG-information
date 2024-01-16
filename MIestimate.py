import classicalEstimators as ee
import numpy as np

def MI(PR=7, n=10000):

    alpha = np.loadtxt('./output/transformed/param_alpha_values.txt')
    PC1a = np.loadtxt('./output/transformed/PC1/PR'+str(PR)+'_out.txt')
    PC2a = np.loadtxt('./output/transformed/PC2/PR'+str(PR)+'_out.txt')
    PC3a = np.loadtxt('./output/transformed/PC3/PR'+str(PR)+'_out.txt')

    #Reconstructs data format using transformed results
    data = np.zeros((n,11))
    data[:,0:8] = alpha[:,0:8]
    data[:,8] = PC1a
    data[:,9] = PC2a
    data[:,10] = PC3a

    print('\nLoaded PR'+str(PR)+':\n')
    DR = data[:,8:11]
    

    #First 9 columns are the parameters, final three are the model output (DR)
    P1 = np.reshape(data[:,0], (-1, 1))
    mi_P1 = abs(ee.MutualInformationKSG(P1, DR, k=5))
    print('MI for P1a: '+str(mi_P1))

    P2 = np.reshape(data[:,1], (-1, 1))
    mi_P2 = abs(ee.MutualInformationKSG(P2, DR, k=5))
    print('MI for P2a: '+str(mi_P2))

    P3 = np.reshape(data[:,2], (-1, 1))
    mi_P3 = abs(ee.MutualInformationKSG(P3, DR, k=5))
    print('MI for P3a: '+str(mi_P3))

    P4 = np.reshape(data[:,3], (-1, 1))
    mi_P4 = abs(ee.MutualInformationKSG(P4, DR, k=5))
    print('MI for P4a: '+str(mi_P4))

    P5 = np.reshape(data[:,4], (-1, 1))
    mi_P5 = abs(ee.MutualInformationKSG(P5, DR, k=5))
    print('MI for P5a: '+str(mi_P5))
    
    P6 = np.reshape(data[:,5], (-1, 1))
    mi_P6 = abs(ee.MutualInformationKSG(P6, DR, k=5))
    print('MI for P6a: '+str(mi_P6))
    
    P7 = np.reshape(data[:,6], (-1, 1))
    mi_P7 = abs(ee.MutualInformationKSG(P7, DR, k=5))
    print('MI for P7a: '+str(mi_P7))
    
    P8 = np.reshape(data[:,7], (-1, 1))
    mi_P8 = abs(ee.MutualInformationKSG(P8, DR, k=5))
    print('MI for P8a: '+str(mi_P8))   
    
    MIs = np.hstack((mi_P1, mi_P2, mi_P3, mi_P4, mi_P5, mi_P6, mi_P7, mi_P8))
    
    return MIs

if __name__ == '__main__':

    for PR in range(1,8):
        MIout = MI(PR)
        np.savetxt('./EE_results/PR'+str(PR)+'_MI.txt', MIout)

