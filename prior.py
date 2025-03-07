import numpy as np
import os

def sample(n=int(1e7)):

	def getV(t):
		A1, A2, A3 = 54, 26, 10
		w1, w2, w3 = 0.007, 0.037, 0.19
		t0 = 2500 

		return -80*(t>=0) - 40*(t>=250) + 40*(t>=300) + 120*(t>=500) - 160*(t>=1500) + 40*(t>=2000) \
		+ 80*(t>=3000) \
		+ (-30 + A1*np.sin(w1*(t-t0)) + A2*np.sin(w2*(t-t0)) + A3*np.sin(w3*(t-t0)))*(t>=3000) \
		- (-30 + A1*np.sin(w1*(t-t0)) + A2*np.sin(w2*(t-t0)) + A3*np.sin(w3*(t-t0)))*(t>=6500) \
		- 120*(t>=6500) + 40*(t>=7500)

	#Each transition rate k is determined by k = A exp(BV), 
	#where A and B are parameters. A and B share the same prior, but not the same actual value.
	def sample_A(n):
		#Params of the form A have a uniform prior between e-7 and 1000 ms-1
		return np.random.uniform(0.0000001, 1000, n)

	def sample_B(n):
		#Params of the form A have a uniform prior between e-7 and 0.4 mV-1
		return np.random.uniform(0.0000001, 0.4, n)

	def sample_GKr(n):
		#Returns random sample of conductance GKr, based on a uniform prior 
		#informed by expfit.py
		return np.random.uniform(0.01748, 0.1748, n)

	def range_check(params, VR):
		#Checks if parameters yeild physiologically possible transition rates (using Pr7) 
		for x in range(len(VR)):
			v = VR[x]
			k1[x,:] = params[0]*np.exp(params[1]*v)
			k2[x,:] = params[2]*np.exp(-params[3]*v)
			k3[x,:] = params[4]*np.exp(params[5]*v)
			k4[x,:] = params[6]*np.exp(-params[7]*v)
			kall = np.concatenate((k1, k2, k3, k4), axis=0)
		return kall
		
	t = np.linspace(0,8000,80000)
	V = getV(t)
	VR = np.array([max(V),min(V)])

	params = np.zeros((11,n))
	
	params[0,:] = sample_A(n)
	params[1,:] = sample_B(n)
	params[2,:] = sample_A(n)
	params[3,:] = sample_B(n)
	params[4,:] = sample_A(n)
	params[5,:] = sample_B(n)
	params[6,:] = sample_A(n)
	params[7,:] = sample_B(n)
	params[8,:] = sample_GKr(n)
	params[9,:] = 1e-8
	params[10,:] = 1e-8

	k1 = np.zeros((len(VR),n))
	k2 = np.zeros((len(VR),n))
	k3 = np.zeros((len(VR),n))
	k4 = np.zeros((len(VR),n))
	
	kvals = range_check(params, VR)
	
	#Only parameters that produce realistic transition rates are used
	pvalid = params[:, np.all((kvals<1000),axis=0)]
	kvalid = kvals[:, np.all((kvals<1000),axis=0)]

	valid = pvalid[:, np.all((kvalid>1.67e-5),axis=0)]

	numgen = np.shape(valid)[1]

	#np.savetxt('params/temp/test_sample.txt', valid)
	
	return valid, numgen
	
if __name__ == '__main__':
	sample(int(1e7))

