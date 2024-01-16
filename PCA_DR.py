import numpy as np
import pr_lengths as prl
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import _pca

def load_data(PR=7, n=10000, tag='NarrowPrior'):
	PRlen = prl.protocol_length(PR)
	data = np.empty((n,PRlen))
	for x in range(n):
		dirname = './output/'+str(n)+str(tag)+'_params/sample_'+str(x)+'/PR'+str(PR)+'_output_'
		data[x,:] = np.load(dirname+str(x)+'.npy')
	return data

def var_find(m=5):
	#Calculates and displays % of explained var for the first n principal components
	var = percent_var[0]
	for x in range(1,m):
		var += percent_var[x]
	print('Explained variance of the first ' + str(m) + ' PCs is: ' + str(var))
	return var

#Edit protocol here:
PR = 7
data = load_data(PR=PR,n=10000,tag='_Alpha')

#First PCA-- exploratory, used to determine sufficient # of PCs to reduce data dimensions to
pca = sklearn.decomposition.PCA(n_components=10)
pca.fit(data[0:10,:])
pca_data = pca.transform(data)

#Plots graph of % explained var for each PC (minimum # of samples of dimensions is used as #PCs) 
percent_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 2)
labels = ['PC' + str(x) for x in range(1, len(percent_var)+1)]
plt.bar(x=range(1, len(percent_var)+1), height = percent_var, tick_label= labels)
plt.xlabel('Principal Component')
plt.ylabel('Percentage Explained Variance')
plt.show()

#Displays explained variance when reduced to lower dimensional space
#nmodes = 5
#var_find(nmodes)

nmodes = 3
var_find(nmodes)

#New PCA object-- fits data to 3 (or optimal #) of PCs and applies transform
pca2 = sklearn.decomposition.PCA(n_components=3)
pca2.fit(data)
DRout = pca2.transform(data)
modes = pca2.components_
svals = pca2.singular_values_

#Writes reduced data and modes to new files
np.savetxt('./output/DR_out/PR'+str(PR)+'_modes.txt', modes)
np.savetxt('./output/DR_out/PR'+str(PR)+'_output.txt', DRout)

