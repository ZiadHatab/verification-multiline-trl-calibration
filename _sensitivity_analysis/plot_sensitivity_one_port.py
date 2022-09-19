import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

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
    s11,s12,s21,s22 = sy.symbols('s11,s12,s21,s22')
    S = sy.Matrix([[s11, s12],[s21, s22]])
    T = S2T(S).applyfunc(sy.simplify)
    
    g = sy.symbols('g')
    R = sy.Matrix([[1, g], [g, 1]])/sy.sqrt(1-g**2) # transformer
            
    SS_ = T2S(R@T@R.inv()).applyfunc(sy.simplify)
    
    SS_  = SS_.subs([(s12,0), (s21,0)]).applyfunc(sy.simplify)
    dSS_ = SS_.diff(g).subs(g,0).applyfunc(sy.simplify)
    SS_  = SS_.subs(g,0).applyfunc(sy.simplify)
    N = 500
    Gnum = np.linspace(-0.99,0.99,N)
    elel = (np.arange(N)+1)/N*2*np.pi
    s21num = np.exp(-1j*elel)
    
    S11 = sy.utilities.lambdify(s11, SS_[0,0])    
    dS11 = sy.utilities.lambdify(s11, dSS_[0,0])
    
    X,Y = np.meshgrid(s21num, Gnum)
    numS11 = S11(X*Y)
    numdS11 = dS11(X*Y)
    
    sens_S11_re = (abs(numS11)/numS11*numdS11).real
    sens_S11_im = (abs(numS11)/numS11*numdS11).imag
    
    
    fig, axs = plt.subplots(1,2, figsize=(5.5, 5.5/2 + 0.2))
    fig.set_dpi(600)
    fig.tight_layout(pad=1.2)
    im = axs[0].imshow(abs(sens_S11_re)/abs(sens_S11_re).max(),
               extent=[round(min(elel)/np.pi),round(max(elel)/np.pi),-1,1],
               aspect = 'auto', vmin=0, vmax=1, cmap='inferno')
    
    axs[1].imshow(abs(sens_S11_im)/abs(sens_S11_im).max(),
               extent=[round(min(elel)/np.pi),round(max(elel)/np.pi),-1,1],
               aspect = 'auto', vmin=0, vmax=1, cmap='inferno')
    
    fig.colorbar(im, ax=axs.ravel().tolist(), location='top',
                 label='Normalized scale')
        
    fig.text(0.5, -0.01, r'Electrical length ($\times \pi$ rad)', ha='center')
    fig.text(-0.015, 0.5, r'Reflection coefficient', va='center', rotation='vertical')
    fig.savefig('1port_sensitivity.pdf', format='pdf', dpi=300, 
                bbox_inches='tight', pad_inches = 0)
    
    plt.show()
    
    # EOF