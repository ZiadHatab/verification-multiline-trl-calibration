'''
compute the uncertianty in Gamma using linear error propagation... 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__=='__main__':
    # nominal impedances
    df_re = pd.read_csv('re_Z0.csv')
    df_im = pd.read_csv('im_Z0.csv')
    f  = df_re.values.T[0]
    Z0 = df_re.values.T[1] + 1j*df_im.values.T[1]
    Z1 = df_re.values.T[2] + 1j*df_im.values.T[2]
    
    # partial er
    df_re = pd.read_csv('re_par_Z0_er.csv')
    df_im = pd.read_csv('im_par_Z0_er.csv')
    par_Z0_er = df_re.values.T[1] + 1j*df_im.values.T[1]
    par_Z1_er = df_re.values.T[2] + 1j*df_im.values.T[2]
    
    # partial h
    df_re = pd.read_csv('re_par_Z0_h.csv')
    df_im = pd.read_csv('im_par_Z0_h.csv')
    par_Z0_h = df_re.values.T[1] + 1j*df_im.values.T[1]
    par_Z1_h = df_re.values.T[2] + 1j*df_im.values.T[2]
    
    # partial t
    df_re = pd.read_csv('re_par_Z0_t.csv')
    df_im = pd.read_csv('im_par_Z0_t.csv')
    par_Z0_t = df_re.values.T[1] + 1j*df_im.values.T[1]
    par_Z1_t = df_re.values.T[2] + 1j*df_im.values.T[2]
    
    # partial w
    df_re = pd.read_csv('re_par_Z0_w.csv')
    df_im = pd.read_csv('im_par_Z0_w.csv')
    par_Z0_w = df_re.values.T[1] + 1j*df_im.values.T[1]
    par_Z1_w = df_re.values.T[2] + 1j*df_im.values.T[2]

    

    JZ0 = np.array([ [[x1.real, x2.real, x3.real, x4.real],
                      [x1.imag, x2.imag, x3.imag, x4.imag]] for x1,x2,x3,x4 in zip(par_Z0_er, par_Z0_h, par_Z0_t, par_Z0_w)])
    JZ1 = np.array([ [[x1.real, x2.real, x3.real, x4.real],
                      [x1.imag, x2.imag, x3.imag, x4.imag]] for x1,x2,x3,x4 in zip(par_Z1_er, par_Z1_h, par_Z1_t, par_Z1_w)])
    
    # data from cross-section images
    h  = np.array([48.82, 47.61, 47.47, 46.99, 46.51, 47.47, 48.43, 46.35, 47.16, 47.95])*1e-3  # in mm
    ws = np.array([209.55, 207.63])*1e-3 # in mm
    wm = np.array([93.51, 93.51, 94.46])*1e-3 # in mm
    t  = np.array([14.69, 14.35, 13.91, 14.39, 12.47, 13.43, 12.95, 10.82, 13.43])*1e-3 # in mm
    
    sigma_er = 0.4
    sigma_h  = 0.002 # np.sqrt( ((h-0.049)**2).mean() )
    sigma_t  = 0.007 # np.sqrt( ((t-0.02)**2).mean() )
    sigma_w  = 0.015 # np.sqrt( ( ((wm-0.107)**2).mean()+ ((ws-0.220)**2).mean() )/2 ) # np.sqrt( ( ws.var(ddof=1) + wm.var(ddof=1) )/2 )
    
    U   = np.diag([sigma_er, sigma_h, sigma_t, sigma_w])**2
    
    UZ0 = np.array([ x@U@x.T for x in JZ0 ])
    UZ1 = np.array([ x@U@x.T for x in JZ1 ])
    
    G  = (Z1 - Z0)/(Z1 + Z0)
    par_G_Z0 = -2*Z1/(Z0 + Z1)**2
    par_G_Z1 =  2*Z0/(Z0 + Z1)**2
    
    JG0 = np.array([ [[ (abs(y)/y*x).real, -(abs(y)/y*x).imag]] for x,y in zip(par_G_Z0, G)])
    JG1 = np.array([ [[ (abs(y)/y*x).real, -(abs(y)/y*x).imag]] for x,y in zip(par_G_Z1, G)])
    
    UG0 = np.array([ x@u@x.T for x,u in zip(JG0,UZ0) ]) + \
            np.array([ x@u@x.T for x,u in zip(JG1,UZ1) ])
        
    UG0 = UG0.squeeze()
    '''
    # write to csv
    data = {'freq (GHz)': f, 'abs(Gamma)': abs(G), 'uabs(Gamma) (var)': UG0}
    df = pd.DataFrame(data)
    df.to_csv('G_with_er_unc.csv', index=False)
    '''
    k = 2
    plt.figure()
    plt.plot(f, abs(G), '-', lw=2, color = 'black')
    plt.plot(f, abs(G) + k*np.sqrt(UG0), '--', color='red', lw=1)
    plt.plot(f, abs(G) - k*np.sqrt(UG0), '--', color='red', lw=1)
    
        
    plt.show()