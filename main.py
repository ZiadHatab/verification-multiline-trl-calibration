import os
import zipfile

# pip install numpy matplotlib scikit-rf metas_unclib scipy -U
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss
import scipy.interpolate as si

# my code
from umTRL import umTRL

import metas_unclib as munc
munc.use_linprop()

def get_cov_component(metas_val, para):
    # To get the uncertainty due to each parameter while accounting for their correlation 
    cov = []
    for inx in range(len(metas_val)):
        J = munc.get_jacobi2(metas_val[inx], para[inx])
        U = munc.get_covariance(para[inx])
        cov.append(J@U@J.T)
    return np.array(cov).squeeze()

def read_waves_to_S_from_zip(zipfile_full_dir, file_name_contain):
    # read wave parameter files and convert to S-parameters (from a zip file)
    with zipfile.ZipFile(zipfile_full_dir, mode="r") as archive:
        netwks = rf.read_zipped_touchstones(archive)
        A = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_A' in key])
        B = rf.NetworkSet([val for key, val in netwks.items() if f'{file_name_contain}_B' in key])    
    freq = A[0].frequency
    S = rf.NetworkSet( [rf.Network(s=b.s@np.linalg.inv(a.s), frequency=freq) for a,b in zip(A,B)] )
    return S.mean_s, S.cov(), np.array([s.s for s in S])

class PlotSettings:
    # to make plots look better for publication
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    def __init__(self, font_size=10, latex=False): 
        self.font_size = font_size 
        self.latex = latex
    def __enter__(self):
        plt.style.use('seaborn-v0_8-paper')
        # make svg output text and not curves
        plt.rcParams['svg.fonttype'] = 'none'
        # fontsize of the axes title
        plt.rc('axes', titlesize=self.font_size*1.2)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=self.font_size)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        # legend fontsize
        plt.rc('legend', fontsize=self.font_size*1)
        # fontsize of the figure title
        plt.rc('figure', titlesize=self.font_size)
        # controls default text sizes
        plt.rc('text', usetex=self.latex)
        #plt.rc('font', size=self.font_size, family='serif', serif='Times New Roman')
        plt.rc('lines', linewidth=1.5)
    def __exit__(self, exception_type, exception_value, traceback):
        plt.style.use('default')

if __name__=='__main__':
    # useful functions
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    path = os.path.dirname(os.path.realpath(__file__)) + '\\'
    
    path_files = path + 'Measurements\\'
    # first cal
    file_name = 'line_50'
    print('Loading files... please wait!!!')
    # these data are already corrected with switch term effects
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path_files + f'{file_name}__0_0mm.zip', f'{file_name}__0_0mm')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path_files + f'{file_name}__0_5mm.zip', f'{file_name}__0_5mm')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path_files + f'{file_name}__1_0mm.zip', f'{file_name}__1_0mm')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path_files + f'{file_name}__3_0mm.zip', f'{file_name}__3_0mm')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path_files + f'{file_name}__5_0mm.zip', f'{file_name}__5_0mm')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path_files + f'{file_name}__6_5mm.zip', f'{file_name}__6_5mm')
    OPEN, OPEN_cov, OPENS = read_waves_to_S_from_zip(path_files + 'open.zip', 'open')
    freq = L1.frequency
    f = freq.f  # frequency axis
    
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [0, 0.5e-3, 1e-3, 3e-3, 5e-3, 6.5e-3]
    ereff_est = 2.5-0.00001j
    reflect = OPEN
    reflect_est = 1
    reflect_offset = -5.3e-3/2 #0.5e-3
    
    # Noise uncertainties
    uSlines   = np.array([L1_cov, L2_cov, L3_cov, L4_cov, L5_cov, L6_cov]) # measured lines
    uSreflect = OPEN_cov # measured reflect 
    
    # length uncertainties
    l_std = 50e-6  # for the line
    ulengths  = l_std**2  # the umTRL code will automatically repeat it for all lines
    
    # mTRL with linear uncertainty evaluation
    cal1 = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est, uSlines=uSlines, 
               uSreflect=uSreflect, ulengths=ulengths)
    cal1.run_umTRL() # run mTRL with linear uncertainty propagation
    
    # plot data and uncertainty
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,4))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0]
        mu  = munc.get_value(cal1.ereff).real
        std = munc.get_stdunc(cal1.ereff).real
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([2, 3])
        #ax.set_yticks(np.arange(4.5, 6.01, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1]
        loss_dbmm_mTRL_model_lin = gamma2dbmm(cal1.gamma)
        mu  = munc.get_value(loss_dbmm_mTRL_model_lin)
        std = munc.get_stdunc(loss_dbmm_mTRL_model_lin)
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 0.3])
        #ax.set_yticks(np.arange(0, 1.51, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
            
        plt.suptitle(r"95% uncertainty bounds ($2\times\sigma$)", 
             verticalalignment='bottom').set_y(0.98)
    
    
    # second cal
    file_name = 'line_30'
    print('Loading files... please wait!!!')
    # these data are already corrected with switch term effects
    L1, L1_cov, L1S = read_waves_to_S_from_zip(path_files + f'{file_name}__0_0mm.zip', f'{file_name}__0_0mm')
    L2, L2_cov, L2S = read_waves_to_S_from_zip(path_files + f'{file_name}__0_5mm.zip', f'{file_name}__0_5mm')
    L3, L3_cov, L3S = read_waves_to_S_from_zip(path_files + f'{file_name}__1_0mm.zip', f'{file_name}__1_0mm')
    L4, L4_cov, L4S = read_waves_to_S_from_zip(path_files + f'{file_name}__3_0mm.zip', f'{file_name}__3_0mm')
    L5, L5_cov, L5S = read_waves_to_S_from_zip(path_files + f'{file_name}__5_0mm.zip', f'{file_name}__5_0mm')
    L6, L6_cov, L6S = read_waves_to_S_from_zip(path_files + f'{file_name}__6_5mm.zip', f'{file_name}__6_5mm')
    OPEN, OPEN_cov, OPENS = read_waves_to_S_from_zip(path_files + 'open.zip', 'open')
    freq = L1.frequency
    f = freq.f  # frequency axis
    
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [0, 0.5e-3, 1e-3, 3e-3, 5e-3, 6.5e-3]
    ereff_est = 2.5-0.00001j
    reflect = OPEN
    reflect_est = 1
    reflect_offset = -5.3e-3/2 #-0.5e-3
    
    # Noise uncertainties
    uSlines   = np.array([L1_cov, L2_cov, L3_cov, L4_cov, L5_cov, L6_cov]) # measured lines
    uSreflect = OPEN_cov # measured reflect 
    
    # length uncertainties
    l_std = 50e-6  # for the line
    ulengths  = l_std**2  # the umTRL code will automatically repeat it for all lines

    # mTRL with linear uncertainty evaluation
    cal2 = umTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=ereff_est,
               uSlines=uSlines, uSreflect=uSreflect,
               ulengths=ulengths)
    cal2.run_umTRL() # run mTRL with linear uncertainty propagation
    
    # plot data and uncertainty
    k = 2 # coverage factor
    with PlotSettings(14):
        fig, axs = plt.subplots(1,2, figsize=(10,4))        
        fig.set_dpi(600)
        fig.tight_layout(pad=2)
        ax = axs[0]
        mu  = munc.get_value(cal2.ereff).real
        std = munc.get_stdunc(cal2.ereff).real
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative effective permittivity')
        ax.set_ylim([2.5, 3.5])
        #ax.set_yticks(np.arange(4.5, 6.01, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        
        ax = axs[1]
        loss_dbmm_mTRL_model_lin = gamma2dbmm(cal2.gamma)
        mu  = munc.get_value(loss_dbmm_mTRL_model_lin)
        std = munc.get_stdunc(loss_dbmm_mTRL_model_lin)
        ax.plot(f*1e-9, mu, lw=2, label='mTRL linear propagation')
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Loss (dB/mm)')
        ax.set_ylim([0, 0.3])
        #ax.set_yticks(np.arange(0, 1.51, 0.3))
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        #ax.legend()
        plt.suptitle(r"95% uncertainty bounds ($2\times\sigma$)", 
             verticalalignment='bottom').set_y(0.98)
    
        
    # load expected Gamma and its uncertainties (derived from 2D HFSS sim)
    path_file = path + '\\csv\\'
    df    = pd.read_csv(path_file + 'Gamma_with_unc.csv')
    fsimm = df.values.T[0]
    G_sim = df.values.T[1] + 1j*df.values.T[2]
    G_sim_cov = np.array([ [[x,y],[y,z]] for x,y,z in zip(df.values.T[3],
                                                          df.values.T[5],
                                                          df.values.T[4])])
    Gsim = np.array([munc.ucomplex(x,covariance=u) for x,u in zip(G_sim,G_sim_cov)])
    k1   = 1 # 68% coverage 
    k2   = 2 # 95% coverage
    
    # in-between error box (the VIAs)    
    X2 = np.array([ munc.ulinalg.dot(munc.ulinalg.inv(x1),x2) for x1,x2 in zip(cal1.X, cal2.X)])
    K2 = cal2.k/cal1.k
    KX2 = np.array([ x2*k2 for x2,k2 in zip(X2,K2)])
    K2 = KX2[:,-1,-1]
    X2 = np.array([ kx2/k2 for kx2,k2 in zip(KX2,K2)])
    
    # T-paramters of the uVIAs
    g11,g21,g12 = X2[:,2,2], X2[:,3,2], X2[:,2,3] # left
    h11,h21,h12 = X2[:,1,1], X2[:,1,3], X2[:,3,1] # right
    
    # left side
    g11 = g11
    g21 = g21
    g12 = g12
    d1 = 0.5e-3
    gamma1 = cal1.gamma
    d2 = 0.5e-3 
    gamma2 = cal2.gamma
    g11,g21,g12 = g11*munc.umath.exp(2*gamma1*d1+2*gamma2*d2), g21*munc.umath.exp(2*gamma2*d2), g12*munc.umath.exp(2*gamma1*d1)
    # model 1: Y-Z
    G1_left = ( (g11 + g21 + g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 + g21 + g12 + 1)**2 + 4*(g11 - g21*g12) )
    # model 2: Z-Y
    G2_left = -( (g11 - g21 - g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 - g21 - g12 + 1)**2 + 4*(g11 - g21*g12) )
    # model 3: symmetric
    G3_left = ( g21 + g12 )/( g11 + 1 )
    
    # right side
    g11 =  h11
    g21 = -h12
    g12 = -h21
    d1 = 0.5e-3
    gamma1 = cal1.gamma
    d2 = 0.5e-3 
    gamma2 = cal2.gamma
    g11,g21,g12 = g11*munc.umath.exp(2*gamma1*d1+2*gamma2*d2), g21*munc.umath.exp(2*gamma2*d2), g12*munc.umath.exp(2*gamma1*d1)
    # model 1: Y-Z
    G1_right = ( (g11 + g21 + g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 + g21 + g12 + 1)**2 + 4*(g11 - g21*g12) )
    # model 2: Z-Y
    G2_right = -( (g11 - g21 - g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 - g21 - g12 + 1)**2 + 4*(g11 - g21*g12) )
    # model 3: symmetric
    G3_right = ( g21 + g12 )/( g11 + 1 )
    
    G1_avg = (G1_right + G1_left)/2
    G2_avg = (G2_right + G2_left)/2
    G3_avg = (G3_right + G3_left)/2
    
    Gammas = [(G1_left, G1_right, G1_avg),
              (G2_left, G2_right, G2_avg),
              (G3_left, G3_right, G3_avg)]
    k = 2 # coverage factor
    for inx, GG in enumerate(Gammas):
        with PlotSettings(14):
            fig, axs = plt.subplots(1,2, figsize=(10,3.5))        
            fig.set_dpi(600)
            fig.tight_layout(pad=2.5)
            ax = axs[0]
            val = abs(GG[0])
            mu  = munc.get_value(val).squeeze()
            std = munc.get_stdunc(val).squeeze()
            ax.plot(f*1e-9, mu, lw=2, label='Left side', marker='>', markersize=12, markevery=30)
            ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
            
            val = abs(GG[1])
            mu  = munc.get_value(val).squeeze()
            std = munc.get_stdunc(val).squeeze()
            ax.plot(f*1e-9, mu, lw=2, label='Right side', marker='<', markersize=12, markevery=30)
            ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.25)
            
            val = abs(GG[2])
            mu  = munc.get_value(val).squeeze()
            std = munc.get_stdunc(val).squeeze()
            ax.plot(f*1e-9, mu, lw=2, label='Average', marker='X', markersize=12, markevery=30)
            ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.2)
            
            val = abs(Gsim)
            mu  = munc.get_value(val).squeeze()
            std = munc.get_stdunc(val).squeeze()
            ax.plot(fsimm, mu, '-', color='black', lw=2.5, label='Simulated')
            ax.plot(fsimm, mu + k1*std, '--', color='black', lw=1.5)
            ax.plot(fsimm, mu - k1*std, '--', color='black', lw=1.5, label='68% coverage')
            ax.plot(fsimm, mu + k2*std, '-.', color='black', lw=1.5)
            ax.plot(fsimm, mu - k2*std, '-.', color='black', lw=1.5, label='95% coverage')
            
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Magnitude (mag)')
            ax.set_ylim([0.15, 0.35])
            ax.set_yticks(np.arange(0.15, 0.351, 0.05))
            ax.set_xlim(0,150)
            ax.set_xticks(np.arange(0,151,30))
            
            ax = axs[1]
            with PlotSettings(8):
                # inset axes....
                axin = ax.inset_axes([0.15, 0.55, 0.4, 0.4])
            
            val = (munc.umath.angle(GG[0]) + 2*np.pi)/np.pi
            mu  = np.unwrap( munc.get_value(val).squeeze(), period=2 )
            std = munc.get_stdunc(val).squeeze()
            p = ax.plot(f*1e-9, mu, lw=2, label='Left side', marker='>', markersize=12, markevery=30)
            q = ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
            axin.plot(f*1e-9, mu, lw=2, label='Left side', marker='>', markersize=12, markevery=30, color=p[0].get_color())
            axin.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3, facecolor=q.get_facecolor())
            
            val = (munc.umath.angle(GG[1]) + 2*np.pi)/np.pi
            mu  = np.unwrap( munc.get_value(val).squeeze(), period=2 )
            std = munc.get_stdunc(val).squeeze()
            p = ax.plot(f*1e-9, mu, lw=2, label='Right side', marker='<', markersize=12, markevery=30)
            q = ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.25)
            axin.plot(f*1e-9, mu, lw=2, label='Right side', marker='<', markersize=12, markevery=30, color=p[0].get_color())
            axin.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.25, facecolor=q.get_facecolor())
            
            val = (munc.umath.angle(GG[2]) + 2*np.pi)/np.pi
            mu  = np.unwrap( munc.get_value(val).squeeze(), period=2 )
            std = munc.get_stdunc(val).squeeze()
            p = ax.plot(f*1e-9, mu, lw=2, label='Average', marker='X', markersize=12, markevery=30)
            q = ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.2)
            axin.plot(f*1e-9, mu, lw=2, label='Average', marker='X', markersize=12, markevery=30, color=p[0].get_color())
            axin.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.2, facecolor=q.get_facecolor())
            
            val = -(munc.umath.angle(Gsim) - 2*np.pi)/np.pi
            mu  = np.unwrap( munc.get_value(val).squeeze(), period=2 )
            std = munc.get_stdunc(val).squeeze()
            ax.plot(fsimm, mu, '-', color='black', lw=2.5, label='Simulated')
            ax.plot(fsimm, mu + k1*std, '--', color='black', lw=1.5)
            ax.plot(fsimm, mu - k1*std, '--', color='black', lw=1.5, label='68% coverage')
            ax.plot(fsimm, mu + k2*std, '-.', color='black', lw=1.5)
            ax.plot(fsimm, mu - k2*std, '-.', color='black', lw=1.5, label='95% coverage')
            
            axin.plot(fsimm, mu, '-', color='black', lw=2.5, label='Simulated')
            axin.plot(fsimm, mu + k1*std, '--', color='black', lw=1.5)
            axin.plot(fsimm, mu - k1*std, '--', color='black', lw=1.5, label='68% coverage')
            axin.plot(fsimm, mu + k2*std, '-.', color='black', lw=1.5)
            axin.plot(fsimm, mu - k2*std, '-.', color='black', lw=1.5, label='95% coverage')
            
            with PlotSettings(8):
                axin.set_xlim((30, 60))
                axin.set_xticks([30,40,50,60])
                axin.set_ylim((0.997, 1.003))
                axin.set_yticks([0.997,0.999,1.001,1.003])
                # axin.set_xticklabels('')
                # axin.set_yticklabels('')
                ax.indicate_inset_zoom(axin, edgecolor="black")
            
            #ax.plot(f_sim*1e-9, mag2db(ntwk_sim.s[:,1,0]), '--' ,lw=3, color='black', label='Simulation')
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel(r'Phase ($\times \pi$ rad)')
            ax.set_ylim([0.9, 1.15])
            ax.set_yticks(np.arange(0.9, 1.151, 0.05))
            ax.set_xlim(0,150)
            ax.set_xticks(np.arange(0,151,30))
            
            if inx < 1:
                with PlotSettings(14):
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.94), 
                               loc='lower center', ncol=3, borderaxespad=0)
            #fig.savefig(path + f'Gamma_model_{inx}.pdf', format='pdf', dpi=300, 
            #                bbox_inches='tight', pad_inches = 0)

    GGsim = si.griddata(fsimm*1e9, munc.get_value(Gsim), f, method='cubic')
    Zpsim  = (1 + GGsim)/(1 - GGsim)
    
    path_file = path + '\\csv\\'
    df_re = pd.read_csv(path_file + 're_Z0.csv')
    df_im = pd.read_csv(path_file + 'im_Z0.csv')
    fsim  = df_re.values.T[0]
    Z1 = df_re.values.T[1] + 1j*df_im.values.T[1]
    Z2 = df_re.values.T[2] + 1j*df_im.values.T[2]
    
    Z1sim = si.griddata(fsim*1e9, Z1, f, method='cubic')
    Z2sim = si.griddata(fsim*1e9, Z2, f, method='cubic')
    
    Zpmeas_left  = (1 + G3_left)/(1 - G3_left)
    Zpmeas_right = (1 + G3_right)/(1 - G3_right)
    
    dZ_left = Z1sim*(Zpmeas_left - Zpsim)
    dZ_right = Z1sim*(Zpmeas_right - Zpsim)
    
    F = ss.savgol_filter(np.eye(len(f)), window_length=9, polyorder=2)  # F is the filtering matrix
    # plot error in impedance with unc budget...
    with PlotSettings(14):
        fig, axs = plt.subplots(2,2, figsize=(10,6.5))        
        fig.set_dpi(600)
        fig.tight_layout(pad=1.5)
        
        ax = axs[0,0]
        val = munc.umath.real(dZ_left)
        mu  = munc.get_value(val).squeeze()
        std = munc.get_stdunc(val).squeeze()
        ax.plot(f*1e-9, mu, lw=2, label='Left transition', marker='<', markersize=12, markevery=30)
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        
        val = munc.umath.real(dZ_right)
        mu  = munc.get_value(val).squeeze()
        std = munc.get_stdunc(val).squeeze()
        ax.plot(f*1e-9, mu, lw=2, label='Right transition', marker='>', markersize=12, markevery=30)
        ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.25)
                
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Real (Ohm)')
        ax.set_yticks(np.arange(-4, 8.1, 2))
        ax.set_ylim([-4, 6])
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        ax.legend(ncol=1, loc='upper left')
        
        ax = axs[0,1]
        with PlotSettings(8):
            # inset axes....
            axin = ax.inset_axes([0.15, 0.55, 0.4, 0.4])
        val = munc.umath.imag(dZ_left)
        mu  = munc.get_value(val).squeeze()
        std = munc.get_stdunc(val).squeeze()
        p = ax.plot(f*1e-9, mu, lw=2, label='Left transition', marker='<', markersize=12, markevery=30)
        q = ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3)
        axin.plot(f*1e-9, mu, lw=2, label='Left transition', marker='<', markersize=12, markevery=30, color=p[0].get_color())
        axin.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.3, facecolor=q.get_facecolor())
        
        val = munc.umath.imag(dZ_right)
        mu  = munc.get_value(val).squeeze()
        std = munc.get_stdunc(val).squeeze()
        p = ax.plot(f*1e-9, mu, lw=2, label='Right transition', marker='>', markersize=12, markevery=30)
        q = ax.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.25)
        axin.plot(f*1e-9, mu, lw=2, label='Right transition', marker='>', markersize=12, markevery=30, color=p[0].get_color())
        axin.fill_between(f*1e-9, mu-std*k, mu+std*k, alpha=0.25, facecolor=q.get_facecolor())
        
        with PlotSettings(8):
            axin.set_xlim((30, 60))
            axin.set_xticks([30,40,50,60])
            axin.set_ylim((-0.2, 0.2))
            axin.set_yticks([-0.2,0,0.2])
            # axin.set_xticklabels('')
            # axin.set_yticklabels('')
            ax.indicate_inset_zoom(axin, edgecolor="black")
            
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Imag (Ohm)')
        ax.set_yticks(np.arange(-4, 8.1, 2))
        ax.set_ylim([-4, 6])
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        # ax.legend(ncol=1, loc='upper left')
        
        ax = axs[1,0]
        val = munc.umath.real(dZ_left)
        std = F.T@np.sqrt(get_cov_component(val, cal1.Slines_metas) + 
                          get_cov_component(val, cal1.Sreflect_metas) + 
                          get_cov_component(val, cal2.Slines_metas) +
                          get_cov_component(val, cal2.Sreflect_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise (left)', marker='<', markersize=12, markevery=30)
        
        val = munc.umath.real(dZ_right)
        std = F.T@np.sqrt(get_cov_component(val, cal1.Slines_metas) + 
                          get_cov_component(val, cal1.Sreflect_metas) + 
                          get_cov_component(val, cal2.Slines_metas) +
                          get_cov_component(val, cal2.Sreflect_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise (right)', marker='>', markersize=12, markevery=30)
        
        val = munc.umath.real(dZ_left)
        std = F.T@np.sqrt(get_cov_component(val, cal1.lengths_metas) + 
                          get_cov_component(val, cal2.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length (left)', marker='v', markersize=12, markevery=30)
        
        val = munc.umath.real(dZ_right)
        std = F.T@np.sqrt(get_cov_component(val, cal1.lengths_metas) + 
                          get_cov_component(val, cal2.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length (right)', marker='^', markersize=12, markevery=30)
        
        val = munc.umath.real(dZ_left)
        std = F.T@munc.get_stdunc(val)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall (left)', linestyle='--', marker='d', markersize=12, markevery=30)
        
        val = munc.umath.real(dZ_right)
        std = F.T@munc.get_stdunc(val)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall (right)', linestyle='--', marker='X', markersize=12, markevery=30)
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Unc(Real) (Ohm)')
        ax.set_yticks(np.arange(0, 6.1, 1))
        ax.set_ylim([0, 5])
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        with PlotSettings(14):
            ax.legend(ncol=1, loc='upper left')
        
        ax = axs[1,1]
        val = munc.umath.imag(dZ_left)
        std = F.T@np.sqrt(get_cov_component(val, cal1.Slines_metas) + 
                          get_cov_component(val, cal1.Sreflect_metas) + 
                          get_cov_component(val, cal2.Slines_metas) +
                          get_cov_component(val, cal2.Sreflect_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise (left)', marker='<', markersize=12, markevery=30)
        
        val = munc.umath.imag(dZ_right)
        std = F.T@np.sqrt(get_cov_component(val, cal1.Slines_metas) + 
                          get_cov_component(val, cal1.Sreflect_metas) + 
                          get_cov_component(val, cal2.Slines_metas) +
                          get_cov_component(val, cal2.Sreflect_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Noise (right)', marker='>', markersize=12, markevery=30)
        
        val = munc.umath.imag(dZ_left)
        std = F.T@np.sqrt(get_cov_component(val, cal1.lengths_metas) + 
                          get_cov_component(val, cal2.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length (left)', marker='v', markersize=12, markevery=30)
        
        val = munc.umath.imag(dZ_right)
        std = F.T@np.sqrt(get_cov_component(val, cal1.lengths_metas) + 
                          get_cov_component(val, cal2.lengths_metas))
        ax.plot(f*1e-9, std*k, lw=2, label='Length (right)', marker='^', markersize=12, markevery=30)
        
        val = munc.umath.imag(dZ_left)
        std = F.T@munc.get_stdunc(val)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall (left)', linestyle='--', marker='d', markersize=12, markevery=30)
        
        val = munc.umath.imag(dZ_right)
        std = F.T@munc.get_stdunc(val)
        ax.plot(f*1e-9, std*k, lw=2, label='Overall (right)', linestyle='--', marker='X', markersize=12, markevery=30)
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Unc(Imag) (Ohm)')
        ax.set_yticks(np.arange(0, 6.1, 1))
        ax.set_ylim([0, 5])
        ax.set_xlim(0,150)
        ax.set_xticks(np.arange(0,151,30))
        # ax.legend(ncol=1, loc='upper left')
        
    #    fig.savefig(path + 'impedance_error_with_unc.pdf', format='pdf', dpi=300, 
    #                    bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    # EOF