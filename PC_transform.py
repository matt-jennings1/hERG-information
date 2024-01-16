import numpy as np
import scipy.optimize as opt
import scipy.stats as stat

def main(PR=7, PC=1, data=None):
    
    def box_cox_s(lam, *args):
        pc = args[0]
        x = data[:,int(pc+8)]
        #x = data
        #Ensures data is positive for inital transform
        if min(x) < 0:
            x += abs(min(x))+0.01
    
        if lam == 0:
            y = np.log(x)
        else:
            y = x**lam
    
        #Set mean to 0 and std to 1
        y = ((y - np.mean(y))/np.std(y))
        y -= np.mean(y)
        s, p = stat.normaltest(y)
        return s
    
    def box_cox(lam, *args):
        pc = args[0]
        x = data[:,int(pc+8)]
        #x = data
        
        #Ensures data is positive and non-zero for inital transform
        if min(x) < 0:
            x += abs(min(x))+0.01
    
        if lam == 0:
            y = np.log(x)
        else:
            y = x**lam
    
        #Set mean to 0 and std to 1
        y = ((y - np.mean(y))/np.std(y))
        y -= np.mean(y)
        return y, lam
    
    f_in = int(PC)
    
    res = opt.minimize(box_cox_s, 0.2, f_in)
    y, lam = box_cox(res['x'], f_in)
    #np.save('./output/transformed/PC'+str(PC)+'/PR'+str(PR)+'_out.npy', y)
    return y

if __name__ == '__main__':
    for PR in range(1,8):
        d_in = np.loadtxt('./output/PR_slices/DR_out/PR'+str(PR)+'_results.txt')
        for PC in range(1,4):
            y = main(PR, PC, d_in)
            np.savetxt('./output/PR_slices/transformed/PC'+str(PC)+'/PR'+str(PR)+'_out.txt', y)
