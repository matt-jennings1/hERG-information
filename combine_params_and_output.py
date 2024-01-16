import numpy as np

n = 10000
tag = '_Alpha'
data_out = np.empty((n,12))

for x in range(n):
    params = np.load('./output/'+str(n)+str(tag+'_params/sample_'+str(x)+'/params_'+str(x)+'.npy')
    data_out[x,:9] = params[:9]

for PR in range(1,8):
        DR = np.loadtxt('./output/DR_out/PR'+str(PR)+'_output.txt')
        data_out[:,9:] = DR
        np.save('./output/DR_out/PR'+str(PR)+'_results.txt', data_out)
