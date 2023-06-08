import sympy as sy
import numpy as np

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

def ABCD2S(A,Z0):
    T = A.copy()
    T[0,0] = A[0,0] + A[0,1]/Z0 - A[1,0]*Z0 - A[1,1]
    T[0,1] = 2*(A[0,0]*A[1,1] - A[0,1]*A[1,0])
    T[1,0] = 2
    T[1,1] = -A[0,0] + A[0,1]/Z0 - A[1,0]*Z0 + A[1,1]
    
    return T/(A[0,0] + A[0,1]/Z0 + A[1,0]*Z0 + A[1,1])

if __name__ == '__main__':
    # shunt element
    y,z = sy.symbols('y,z')
    s11, s21 = sy.symbols('s11, s21')
    q = sy.symbols('q')
    q = 1
    shunt = ABCD2S(sy.Matrix([[1,0],[y,1]]), q)
    series = ABCD2S(sy.Matrix([[1,z],[0,1]]), q)
    Tsh = S2T(shunt).applyfunc(sy.simplify)
    Ter = S2T(series).applyfunc(sy.simplify)
    Ts  = S2T(sy.Matrix([[s11, s21],[s21, s11]])) # Y:S11; Z:S21
    
    # parasitic models
    # model 1: Y-Z-Q
    M1 = (Tsh@Ter).applyfunc(sy.simplify)
    # model 2: Z-Y-Q
    M2 = (Ter@Tsh).applyfunc(sy.simplify)
    # model 3: S-Q  symetric network
    M3 = (Ts).applyfunc(sy.simplify)
    
    # Transformer
    gam = sy.symbols('Gamma')
    R = sy.Matrix([[1, gam],[gam, 1]])/(sy.sqrt(1-gam**2))
    
    # line offsets
    t1, t2 = sy.symbols('t1 t2')
    L1 = sy.Matrix([[t1, 0],[0, 1/t1]])
    L2 = sy.Matrix([[t2, 0],[0, 1/t2]])
    
    # measured transition
    g11, g12, g21 = sy.symbols('g11, g12, g21')
    G = sy.Matrix([[g11, g12],[g21, 1]])
        
    MM1 = (M1@R).applyfunc(sy.simplify)
    MM2 = (M2@R).applyfunc(sy.simplify)
    MM3 = (M3@R).applyfunc(sy.simplify)
        
    sol1 = sy.solve((MM1/MM1[-1,-1] - G).vec()[:-1], [gam, y, z], 
                   simplify=True, manual=False, doit=False)
    
    sol2 = sy.solve((MM2/MM2[-1,-1] - G).vec()[:-1], [gam, y, z], 
                   simplify=True, manual=False, doit=False)
    
    sol3 = sy.solve((MM3/MM3[-1,-1] - G).vec()[:-1], [gam, s11, s21**2], 
                   simplify=True, manual=False, doit=False)
    
    ## the obtained models are:
    # model 1
    G1 = ( (g11 + g21 + g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 + g21 + g12 + 1)**2 + 4*(g11 - g21*g12) )
    y1 = (-g11 + g21 - g12 + 1)/(g11 + g21 + g12 + 1)
    z1 = ((g12 + 1)**2 - (g11 + g21)**2)/(g11 - g21*g12)/4

    # model 2
    G2 = -( (g11 - g21 - g12 + 1)**2 - 4*(g11 - g21*g12) )/( (g11 - g21 - g12 + 1)**2 + 4*(g11 - g21*g12) )
    y2 = ((g12 - 1)**2 - (g11 - g21)**2)/(g11 - g21*g12)/4 
    z2 = (-g11 - g21 + g12 + 1)/(g11 - g21 - g12 + 1)

    # model 3
    G3 = ( g21 + g12 )/( g11 + 1 )
    tt = (g11 - g21*g12)*( (g11 +1)**2 - (g21 + g12)**2 )/(g11 - g21*g12 - g21**2 + 1)**2
    r  = (g12 - g11*g21)/(g11 - g21*g12 - g21**2 + 1)

    
# EOF