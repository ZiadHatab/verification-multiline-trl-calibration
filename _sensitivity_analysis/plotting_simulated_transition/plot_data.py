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
    
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5.5/1.5))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    ax.plot(f*1e-9, abs(G1), '>-', lw=2, label='Model 1', markevery=10, markersize=8)
    ax.plot(f*1e-9, abs(G2), '<-', lw=2, label='Model 2', markevery=10, markersize=8)
    ax.plot(f*1e-9, abs(G3), '^-', lw=2, label='Model 3', markevery=10, markersize=8)
    ax.plot(f*1e-9, abs(G1_off), 's--', lw=2, markevery=10, markersize=8,
             label='Model 1 with +30um offset error')
    ax.plot(f*1e-9, abs(G2_off), 'o--', lw=2, markevery=10, markersize=8,
             label='Model 2 with +30um offset error')
    ax.plot(f*1e-9, abs(G3_off), 'v--', lw=2, markevery=10, markersize=8,
             label='Model 3 with +30um offset error')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Reflection coefficient (mag)')
    plt.legend(ncol=2)
    ax.set_ylim([0.22, 0.26])
    ax.set_yticks(0.22 + np.arange(5)*0.01)
    ax.set_xlim([0, 150])
    ax.set_xticks(np.arange(6)*30)
    fig.savefig('simulated_transition.pdf', format='pdf', dpi=300, 
                    bbox_inches='tight', pad_inches = 0)
    
    g11 = g11_
    g21 = g21_
    g12 = g12_
    
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
    plt.plot(y1.imag)
    plt.plot(y2.imag)
    
    plt.figure()
    plt.plot(tt.imag)
    #plt.plot(r.imag)
    
    plt.show()

# EOF