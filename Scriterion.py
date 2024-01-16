import numpy as np

for PR in range(1,8):

    MI = np.loadtxt('./EE_results/PR'+str(PR)+'_MI.txt')
    temp = np.loadtxt('./EE_results/PR'+str(PR)+'_CMI_matrix.txt').flatten()
    CMI = temp[~np.isnan(temp)]
    nCMI = len(CMI)

    S1 = (1/8) * np.sum(MI) - (1/nCMI) * np.sum(CMI)
    S2 = np.std(MI)
    S3 = (1/8) * np.sum(MI) - (1/nCMI) * np.sum(CMI) - np.std(MI)

    print("S1-3 for Pr"+str(PR)+":")
    print(S1)
    print(S2)
    print(str(S3)+"\n")
    continue



