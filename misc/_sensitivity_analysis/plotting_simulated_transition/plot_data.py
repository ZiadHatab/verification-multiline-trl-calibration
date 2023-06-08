import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

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

if __name__=='__main__':
    
    ntwk = rf.Network('hfss_impedance_transition.s2p')
    gamma1 = ntwk.gamma[:,0]
    gamma2 = ntwk.gamma[:,1]
    f = ntwk.frequency.f
    
    T     = np.array([S2T(x) for x in ntwk.s])
    Tnorm = np.array([x/x[-1,-1] for x in T])
    
    g11 = np.array([x[0,0] for x in Tnorm])
    g21 = np.array([x[1,0] for x in Tnorm])
    g12 = np.array([x[0,1] for x in Tnorm])
    
    d1  = 0.103e-3
    d2  = 0.103e-3
    
    exp1 = np.exp(2*gamma1*d1)
    exp2 = np.exp(2*gamma2*d2)
    g11_ = g11*exp1*exp2
    g21_ = g21*exp2
    g12_ = g12*exp1

    # model 1: Y-Z
    G1 = ( (g11_ + g21_ + g12_ + 1)**2 - 4*(g11_ - g21_*g12_) )/( (g11_ + g21_ + g12_ + 1)**2 + 4*(g11_ - g21_*g12_) )
    # model 2: Z-Y
    G2 = -( (g11_ - g21_ - g12_ + 1)**2 - 4*(g11_ - g21_*g12_) )/( (g11_ - g21_ - g12_ + 1)**2 + 4*(g11_ - g21_*g12_) )
    # model 3: symmetric
    G3 = ( g21_ + g12_ )/( g11_ + 1 )
    
    # offset error versions
    exp1 = np.exp(2*gamma1*(d1+0.0e-3))
    exp2 = np.exp(2*gamma2*(d2+0.03e-3))
    g11_ = g11*exp1*exp2
    g21_ = g21*exp2
    g12_ = g12*exp1
    
    # model 1: Y-Z
    G1_off = ( (g11_ + g21_ + g12_ + 1)**2 - 4*(g11_ - g21_*g12_) )/( (g11_ + g21_ + g12_ + 1)**2 + 4*(g11_ - g21_*g12_) )
    # model 2: Z-Y
    G2_off = -( (g11_ - g21_ - g12_ + 1)**2 - 4*(g11_ - g21_*g12_) )/( (g11_ - g21_ - g12_ + 1)**2 + 4*(g11_ - g21_*g12_) )
    # model 3: symmetric
    G3_off = ( g21_ + g12_ )/( g11_ + 1 )
    '''
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5/1.5))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    '''
    # GG = (ntwk.z0[:,1] - ntwk.z0[:,0])/(ntwk.z0[:,1] + ntwk.z0[:,0])
    
    fig, axs = plt.subplots(1,2, figsize=(5.5, 5.5/2))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.5)
    ax = axs[0]
    ax.plot(f*1e-9, abs((g21_+g12_)/2), '>-', lw=2, label='Without Modeling', markevery=10, markersize=10)
    ax.plot(f*1e-9, abs(G1), 'v-', lw=2, label='Model 1', markevery=10, markersize=10)
    ax.plot(f*1e-9, abs(G2), '>-', lw=2, label='Model 2', markevery=10, markersize=10)
    ax.plot(f*1e-9, abs(G3), '<-', lw=2, label='Model 3', markevery=10, markersize=10)
    ax.plot(f*1e-9, abs(G1_off), 'h--', lw=2, markevery=10, markersize=8,
             label='Model 1 with +30um offset')
    ax.plot(f*1e-9, abs(G2_off), 'o--', lw=2, markevery=10, markersize=8,
             label='Model 2 with +30um offset')
    ax.plot(f*1e-9, abs(G3_off), 'X--', lw=2, markevery=10, markersize=8,
             label='Model 3 with +30um offset')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Reflection (mag)')
    ax.set_yticks(np.arange(0.18,0.31,0.02))
    ax.set_ylim([0.2, 0.26])
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    
    ax = axs[1]
    ax.plot(f*1e-9, np.unwrap(np.angle((g21_+g12_)/2) + 2*np.pi)/np.pi, '>-', lw=2, label='Without Modeling', markevery=10, markersize=10)
    ax.plot(f*1e-9, np.unwrap(np.angle(G1))/np.pi, 'v-', lw=2, label='Model 1', markevery=10, markersize=10)
    ax.plot(f*1e-9, np.unwrap(np.angle(G2))/np.pi, '>-', lw=2, label='Model 2', markevery=10, markersize=10)
    ax.plot(f*1e-9, np.unwrap(np.angle(G3))/np.pi, '<-', lw=2, label='Model 3', markevery=10, markersize=10)
    ax.plot(f*1e-9, np.unwrap(np.angle(G1_off))/np.pi, 'h--', lw=2, markevery=10, markersize=8,
             label='Model 1 w/ +30um offset')
    ax.plot(f*1e-9, np.unwrap(np.angle(G2_off))/np.pi, 'o--', lw=2, markevery=10, markersize=8,
             label='Model 2 w/ +30um offset')
    ax.plot(f*1e-9, np.unwrap(np.angle(G3_off))/np.pi, 'X--', lw=2, markevery=10, markersize=8,
             label='Model 3 w/ +30um offset')
    
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(r'Reflection (phase $\times \pi$)')
    ax.set_yticks(np.arange(0.1,2,0.05))
    ax.set_ylim([0.95, 1.1])
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    
    # plt.legend(ncol=2, loc='upper right')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.96), 
                   loc='lower center', ncol=3, borderaxespad=0, 
                   columnspacing=1.5, fontsize=8)
    fig.savefig('simulated_transition.pdf', format='pdf', dpi=300, 
                    bbox_inches='tight', pad_inches = 0)
        
    plt.show()

# EOF