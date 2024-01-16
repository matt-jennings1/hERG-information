import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

def main(PR=7, plot=False):
    #Helper function
    def log_transform(x, x0):
        return np.log((x/x0))
    
    data = np.loadtxt('./output/DR_out/PR'+str(PR)+'/PR'+str(PR)+'_results.txt')
    params = data[:,0:8] #index 8 is GKR-- omitted
    GKr = data[:,8]
    del(data)
    
    P0_1 = np.mean(params[:,0])
    P0_2 = np.mean(params[:,1])
    P0_3 = np.mean(params[:,2])
    P0_4 = np.mean(params[:,3])
    P0_5 = np.mean(params[:,4])
    P0_6 = np.mean(params[:,5])
    P0_7 = np.mean(params[:,6])
    P0_8 = np.mean(params[:,7])
    
    a_1 = log_transform(params[:,0], P0_1)
    a_2 = log_transform(params[:,1], P0_2)
    a_3 = log_transform(params[:,2], P0_3)
    a_4 = log_transform(params[:,3], P0_4)
    a_5 = log_transform(params[:,4], P0_5)
    a_6 = log_transform(params[:,5], P0_6)
    a_7 = log_transform(params[:,6], P0_7)
    a_8 = log_transform(params[:,7], P0_8)
    
    skew_1 = stat.skew(a_1)
    skew_2 = stat.skew(a_2)
    skew_3 = stat.skew(a_3)
    skew_4 = stat.skew(a_4)
    skew_5 = stat.skew(a_5)
    skew_6 = stat.skew(a_6)
    skew_7 = stat.skew(a_7)
    skew_8 = stat.skew(a_8)

    if plot == True:
        fig, ax = plt.subplots(3,3)
        ax[0,0].hist(a_1, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[0,0].set_title('Distribution of alpha-1', fontsize='medium')
                                                                                           
        ax[0,1].hist(a_2, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[0,1].set_title('Distribution of alpha-2', fontsize='medium')
                                                                                           
        ax[0,2].hist(a_3, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[0,2].set_title('Distribution of alpha-3', fontsize='medium')
                                                                                           
        ax[1,0].hist(a_4, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[1,0].set_title('Distribution of alpha-4', fontsize='medium')
                                                                                           
        ax[1,1].hist(a_5, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[1,1].set_title('Distribution of alpha-5', fontsize='medium')
                                                                                           
        ax[1,2].hist(a_6, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[1,2].set_title('Distribution of alpha-6', fontsize='medium')
                                                                                           
        ax[2,0].hist(a_7, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[2,0].set_title('Distribution of alpha-7', fontsize='medium')
                                                                                           
        ax[2,1].hist(a_8, bins = np.linspace(-15,5,100), edgecolor='black')
        ax[2,1].set_title('Distribution of alpha-8', fontsize='medium')
                                                                                           
        ax[2,2].hist(GKr, bins = 100, edgecolor = 'black')
        ax[2,2].set_title('Distribution of GKr (non-transformed)', fontsize='medium')    
        
        plt.show()
    
    output = np.array((a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,GKr))
    
    np.savetxt('./output/transformed/param_alpha_values.txt', np.transpose(output))
    np.savetxt('./output/transformed/param_means.txt', np.vstack((P0_1, P0_2, P0_3, P0_4, P0_5, P0_6, P0_7, P0_8))) 
    print('Completed transformations')

if __name__ == '__main__':
    
    #The protocol used here is unimportant, as each results file should contain the same parameters
    main(PR=7, plot=True)

