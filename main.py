"""
mTRL verification
@author: Ziad Hatab (zi.hatab@gmail.com)
"""
import os  # part of python standard library

import skrf as rf      # for RF stuff
import numpy as np
import matplotlib.pyplot as plt   # for plotting
import pandas as pd

# my script (MultiCal.py and TUGmTRL must also be in same folder)
from mTRL import mTRL

def S2T(S):
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    
    return T/S[1,0]

def T2S(T):
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    
    return S/T[1,1]

# main script
if __name__ == '__main__':
    c0 = 299792458   # speed of light in vacuum (m/s)
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    alpha2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))
    
    f_low = 2   # in GHz
    f_up  = 150 # in GHz
    
    # load switch term (used for both calibration)
    # files' path are reference to script's path
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\s2p\\RAW\\'
    # the order got mixed
    gamma_f = rf.Network(s2p_path + 'switch_terms.s2p').s12[f'{f_low}ghz-{f_up}ghz']
    gamma_r = rf.Network(s2p_path + 'switch_terms.s2p').s21[f'{f_low}ghz-{f_up}ghz']
    
    # load first mTRL data (reference mTRL)
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\s2p\\RAW\\copper2_match\\'
    # Calibration standards
    L1    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_line_0_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L2    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_line_0_5mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L3    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_line_1_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L4    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_line_3_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L5    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_line_5_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L6    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_line_6_5mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    SHORT = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_short_0_5mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    f = L1.frequency.f
    line_dut = L4
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [0, 0.5e-3, 1e-3, 3e-3, 5e-3, 6.5e-3]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [0.5e-3]
    
    cal_ref = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=2.4+0j, switch_term=[gamma_f, gamma_r])
        
    # using TUG mTRL
    cal_ref.run_tug()
    gamma_ref = cal_ref.gamma
    ereff_ref = cal_ref.ereff
    
    # load second mTRL data (verification mTRL)
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\s2p\\RAW\\copper2_step\\'
    # Calibration standards
    L1    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_step_0_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L2    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_step_0_5mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L3    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_step_1_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L4    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_step_3_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L5    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_step_5_0mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    L6    = rf.NetworkSet(rf.read_all(s2p_path, contains='RAW_step_6_5mm')).mean_s[f'{f_low}ghz-{f_up}ghz']
    f = L1.frequency.f
    step_dut = L4
    lines = [L1, L2, L3, L4, L5, L6]
    line_lengths = [0, 0.5e-3, 1e-3, 3e-3, 5e-3, 6.5e-3]
    reflect = [SHORT]
    reflect_est = [-1]
    reflect_offset = [-0.5e-3]
    
    cal_verf = mTRL(lines=lines, line_lengths=line_lengths, reflect=reflect, 
               reflect_est=reflect_est, reflect_offset=reflect_offset, 
               ereff_est=2.4+0j, switch_term=[gamma_f, gamma_r])
        
    # using TUG mTRL
    cal_verf.run_tug()
    gamma_verf = cal_verf.gamma
    ereff_verf = cal_verf.ereff
    
    fig, axs = plt.subplots(1,2, figsize=(5.5, 5.5/2))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    ax = axs[0]
    ax.plot(f*1e-9, alpha2dbmm(gamma_ref.real), '^-', lw=2, label='Primary mTRL', 
            markevery=80, markersize=8)
    ax.plot(f*1e-9, alpha2dbmm(gamma_verf.real), 'v-', lw=2, label='Verification mTRL (step lines)', 
            markevery=80, markersize=8)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Attenuation (dB/mm)')
    ax.set_ylim([0, 0.3])
    ax.set_yticks(np.arange(6)*0.06)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    
    ax = axs[1]
    ax.plot(f*1e-9, ereff_ref.real, '^-', lw=2, label='Primary mTRL', 
            markevery=80, markersize=8)
    ax.plot(f*1e-9, ereff_verf.real, 'v-', lw=2, label='Verification mTRL (step lines)', 
            markevery=80, markersize=8)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Effective relative permittivity')
    ax.set_ylim([2, 3])
    ax.set_yticks(np.arange(6)*0.2 + 2)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.96), 
                   loc='lower center', ncol=3, borderaxespad=0)
    #fig.savefig('losses_and_ereff.pdf', format='pdf', dpi=300, 
    #            bbox_inches='tight', pad_inches = 0)
    
    
    # plot calibrated lines (line 3mm)
    line_cal = cal_ref.apply_cal(line_dut)
    cal_ref.shift_plane(0.5e-3)
    step_cal = cal_ref.apply_cal(step_dut)
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5/1.5))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    ax.plot(f*1e-9, mag2db(line_cal.s21.s).squeeze(), '^-', lw=2, 
            label='Calibrated 3mm line', markevery=60, markersize=10)
    ax.plot(f*1e-9, mag2db(step_cal.s21.s).squeeze(), 'v-', lw=2, 
            label='Calibrated 3mm step line', markevery=60, markersize=10)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S21 (dB)')
    plt.legend(ncol=1)
    ax.set_ylim([-2.5, 0.5])
    ax.set_yticks(-2.5 + np.arange(7)*0.5)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    cal_ref.shift_plane(-0.5e-3) # undo the shift
    #fig.savefig('calibrated_line.pdf', format='pdf', dpi=300, 
    #            bbox_inches='tight', pad_inches = 0)
    
    # plot calibrated SHORT
    SHORT_cal = cal_ref.apply_cal(SHORT)
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5/1.5))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    ax.plot(f*1e-9, mag2db(SHORT_cal.s11.s).squeeze(), '^-', lw=2, 
            label='Calibrated offset micro-VIA short', markevery=60, markersize=10)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S11 (dB)')
    plt.legend(ncol=2)
    ax.set_ylim([-0.8, 0.4])
    ax.set_yticks(-0.8 + np.arange(7)*0.2)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    #fig.savefig('calibrated_short.pdf', format='pdf', dpi=300, 
    #        bbox_inches='tight', pad_inches = 0)
    
    # impedance transition (inbetween error-boxes)
    # same as the paper, but using the kroneker product definition of error-boxes
    X = np.linalg.pinv(cal_ref.X)@cal_verf.X
    k = 1/cal_ref.K*cal_verf.K
    kX = np.array([ x2*k2 for x2,k2 in zip(X,k)])
    k = kX[:,-1,-1]
    X = np.array([ kx2/k2 for kx2,k2 in zip(kX,k)])
    
    g11,g21,g12 = X[:,2,2], X[:,3,2], X[:,2,3]  
    h11,h21,h12 = X[:,1,1], X[:,1,3], X[:,3,1]
    
    # Average
    #g11 = (g11 + h11)/2
    #g21 = (g21 - h12)/2
    #g12 = (g12 - h21)/2
    
    # left side
    g11 = g11
    g21 = g21
    g12 = g12
    d1 = 0.5e-3
    gamma1 = gamma_ref
    d2 = 0.5e-3 
    gamma2 = gamma_verf
    g11,g21,g12 = g11*np.exp(2*gamma1*d1+2*gamma2*d2), g21*np.exp(2*gamma2*d2), g12*np.exp(2*gamma1*d1)
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
    gamma1 = gamma_ref
    d2 = 0.5e-3 
    gamma2 = gamma_verf
    g11,g21,g12 = g11*np.exp(2*gamma1*d1+2*gamma2*d2), g21*np.exp(2*gamma2*d2), g12*np.exp(2*gamma1*d1)
    # model 1: Y-Z
    G1_right = ( (g11 + g21 + g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 + g21 + g12 + 1)**2 + 4*(g11 - g21*g12) )
    # model 2: Z-Y
    G2_right = -( (g11 - g21 - g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 - g21 - g12 + 1)**2 + 4*(g11 - g21*g12) )
    # model 3: symmetric
    G3_right = ( g21 + g12 )/( g11 + 1 )
    
    
    # load expected Gamma and its uncertainties (derived from 2D HFSS sim)
    s2p_path = os.path.dirname(os.path.realpath(__file__)) + '\\csv\\'
    df    = pd.read_csv(s2p_path + 'G_with_all_unc.csv')
    fsimm = df.values.T[0]
    G_sim = df.values.T[1]
    G_sim_unc = np.sqrt(df.values.T[2])
    k1     = 1    # 68% coverage 
    k2     = 1.96 # 95% coverage
    
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5/1.5))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    ax.plot(fsimm, G_sim, '-', color='black', lw=2.5, label='Expected')
    ax.plot(fsimm, G_sim + k1*G_sim_unc, '--', color='black', lw=1.5)
    ax.plot(fsimm, G_sim - k1*G_sim_unc, '--', color='black', lw=1.5, label='68% coverage')
    ax.plot(fsimm, G_sim + k2*G_sim_unc, '-.', color='black', lw=1.5)
    ax.plot(fsimm, G_sim - k2*G_sim_unc, '-.', color='black', lw=1.5, label='95% coverage')
    
    ax.plot(f*1e-9, abs(G1_left), '>-', lw=2, label='Model 1 (left)', markevery=60, markersize=10)
    ax.plot(f*1e-9, abs(G2_left), '<-', lw=2, label='Model 2 (left)', markevery=60, markersize=10)
    ax.plot(f*1e-9, abs(G3_left), '^-', lw=2, label='Model 3 (left)', markevery=60, markersize=10)
    
    ax.plot(f*1e-9, abs(G1_right), 's-', lw=2, label='Model 1 (right)', markevery=60, markersize=10)
    ax.plot(f*1e-9, abs(G2_right), 'o-', lw=2, label='Model 2 (right)', markevery=60, markersize=10)
    ax.plot(f*1e-9, abs(G3_right), 'P-', lw=2, label='Model 3 (right)', markevery=60, markersize=10)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Reflection coefficient (mag)')
    plt.legend(ncol=3, columnspacing=1)
    ax.set_ylim([0.1, 0.4])
    ax.set_yticks(0.1 + np.arange(7)*0.05)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    #fig.savefig('impedance_transition_verification_2.pdf', format='pdf', dpi=300, 
    #        bbox_inches='tight', pad_inches = 0)
    
    
    df    = pd.read_csv(s2p_path + 'G_with_er_unc.csv')
    fsimm = df.values.T[0]
    G_sim_unc_er = np.sqrt(df.values.T[2])
    df    = pd.read_csv(s2p_path + 'G_with_h_unc.csv')
    fsimm = df.values.T[0]
    G_sim_unc_h = np.sqrt(df.values.T[2])
    df    = pd.read_csv(s2p_path + 'G_with_t_unc.csv')
    fsimm = df.values.T[0]
    G_sim_unc_t = np.sqrt(df.values.T[2])
    df    = pd.read_csv(s2p_path + 'G_with_w_unc.csv')
    fsimm = df.values.T[0]
    G_sim_unc_w = np.sqrt(df.values.T[2])
    
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5/1.5))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    ax.plot(fsimm, G_sim_unc_er, '>-', lw=2, label='Permittivity', markevery=10, markersize=10)
    ax.plot(fsimm, G_sim_unc_h, '<-', lw=2, label='Substrate thickness', markevery=10, markersize=10)
    ax.plot(fsimm, G_sim_unc_t, '^-', lw=2, label='Trace thickness', markevery=10, markersize=10)
    ax.plot(fsimm, G_sim_unc_w, 'v-', lw=2, label='Trace width', markevery=10, markersize=10)
    ax.plot(fsimm, G_sim_unc, 'o-', lw=2, label='All contributions', markevery=10, markersize=10)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Reflection coefficient uncertainty (mag)')
    plt.legend(ncol=2)
    ax.set_ylim([0, 0.06])
    ax.set_yticks(np.arange(7)*0.01)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    #fig.savefig('Gamma_uncertainties.pdf', format='pdf', dpi=300, 
    #        bbox_inches='tight', pad_inches = 0)
    
    '''
    ## The parasitics for those interested...
    # model 1
    y1 = (-g11 + g21 - g12 + 1)/(g11 + g21 + g12 + 1)
    z1 = ((g12 + 1)**2 - (g11 + g21)**2)/(g11 - g21*g12)/4

    # model 2
    y2 = ((g12 - 1)**2 - (g11 - g21)**2)/(g11 - g21*g12)/4 
    z2 = (-g11 - g21 + g12 + 1)/(g11 - g21 - g12 + 1)

    # model 3
    tt = (g11 - g21*g12)*( (g11 +1)**2 - (g21 + g12)**2 )/(g11 - g21*g12 - g21**2 + 1)**2
    r  = (g12 - g11*g21)/(g11 - g21*g12 - g21**2 + 1)
    
    plt.figure()
    plt.plot(z1.imag)
    plt.plot(z2.imag)
    '''

    plt.show()
    
# EOF