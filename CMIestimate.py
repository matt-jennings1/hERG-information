from npeet import entropy_estimators as ee
import numpy as np

def CMI(Pr=7):

    data = np.zeros((10000,11))
    data[:,0:8] = np.loadtxt('./output/transformed/param_alpha_values.txt')[:,0:8]
    data[:,8] = np.loadtxt('./output/transformed/PC1/PR'+str(Pr)+'_out.txt')
    data[:,9] = np.loadtxt('./output/transformed/PC2/PR'+str(Pr)+'_out.txt')
    data[:,10] = np.loadtxt('./output/transformed/PC3/PR'+str(Pr)+'_out.txt')

    DR = data[:,8:]

    #First 9 columns are the parameters, final three are the model output (DR)
    P1 = np.reshape(data[:,0], (-1, 1))

    P2 = np.reshape(data[:,1], (-1, 1))

    P3 = np.reshape(data[:,2], (-1, 1))

    P4 = np.reshape(data[:,3], (-1, 1))

    P5 = np.reshape(data[:,4], (-1, 1))
    
    P6 = np.reshape(data[:,5], (-1, 1))
    
    P7 = np.reshape(data[:,6], (-1, 1))
    
    P8 = np.reshape(data[:,7], (-1, 1))

    CMIs = np.empty((7,8))
    
    #Note: there is almost certainly a more efficient way of doing this

    CMIs[0,0] = None
    CMIs[0,1] = ee.mi(P2, DR, P1)
    CMIs[0,2] = ee.mi(P3, DR, P1)
    CMIs[0,3] = ee.mi(P4, DR, P1)
    CMIs[0,4] = ee.mi(P5, DR, P1)
    CMIs[0,5] = ee.mi(P6, DR, P1)
    CMIs[0,6] = ee.mi(P7, DR, P1)
    CMIs[0,7] = ee.mi(P8, DR, P1)

    CMIs[1,0] = None
    CMIs[1,1] = None
    CMIs[1,2] = ee.mi(P3, DR, P2)
    CMIs[1,3] = ee.mi(P4, DR, P2)
    CMIs[1,4] = ee.mi(P5, DR, P2)
    CMIs[1,5] = ee.mi(P6, DR, P2)
    CMIs[1,6] = ee.mi(P7, DR, P2)
    CMIs[1,7] = ee.mi(P8, DR, P2)

    CMIs[2,0] = None 
    CMIs[2,1] = None
    CMIs[2,2] = None
    CMIs[2,3] = ee.mi(P4, DR, P3)
    CMIs[2,4] = ee.mi(P5, DR, P3)
    CMIs[2,5] = ee.mi(P6, DR, P3)
    CMIs[2,6] = ee.mi(P7, DR, P3)
    CMIs[2,7] = ee.mi(P8, DR, P3)

    CMIs[3,0] = None
    CMIs[3,1] = None
    CMIs[3,2] = None
    CMIs[3,3] = None
    CMIs[3,4] = ee.mi(P5, DR, P4)
    CMIs[3,5] = ee.mi(P6, DR, P4)
    CMIs[3,6] = ee.mi(P7, DR, P4)
    CMIs[3,7] = ee.mi(P8, DR, P4)

    CMIs[4,0] = None
    CMIs[4,1] = None
    CMIs[4,2] = None
    CMIs[4,3] = None
    CMIs[4,4] = None
    CMIs[4,5] = ee.mi(P6, DR, P5)
    CMIs[4,6] = ee.mi(P7, DR, P5)
    CMIs[4,7] = ee.mi(P8, DR, P5)

    CMIs[5,0] = None
    CMIs[5,1] = None
    CMIs[5,2] = None
    CMIs[5,3] = None
    CMIs[5,4] = None
    CMIs[5,5] = None
    CMIs[5,6] = ee.mi(P7, DR, P6)
    CMIs[5,7] = ee.mi(P8, DR, P6)

    CMIs[6,0] = None
    CMIs[6,1] = None
    CMIs[6,2] = None
    CMIs[6,3] = None
    CMIs[6,4] = None
    CMIs[6,5] = None
    CMIs[6,6] = None
    CMIs[6,7] = ee.mi(P8, DR, P7)

    np.savetxt('./EE_results/PR'+str(Pr)+'_CMI_matrix.txt', CMIs)
    print('\nFinished CMI for PR'+str(Pr))

    return CMIs

if __name__ == '__main__':
    for PR in range(1,8):
        out = CMI(PR)
