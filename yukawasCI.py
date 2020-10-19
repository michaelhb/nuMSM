import numpy as np
from numpy import pi, cos, sin, exp

def FM(M, dM, Imw, Rew, delta, eta,  H = 1, xi = 1):
    '''returns an array of Yukawas in the mass basis
    this has been validated against the mathematica code'''

    v0 = 174.1
    w = Rew + 1.0j*Imw
    cw, sw = cos(w), sin(w)

    M2, M3 = M-dM, M+dM
    #  note the precision!
    MN = np.sqrt([[M2,0], [0,M3]])

    if H == 1:
        #NH case  
        th12, th13, th23  = (np.pi/180.0)*np.array([33.62, 8.54, 47.2])
        msol, matm = np.sqrt([7.40e-5*1e-18, 2.494e-3*1e-18])
        m1, m2, m3 = 0.0, msol, matm  #NH

        Omega = np.matrix([[0, 0],[cw, sw],[-xi*sw, xi*cw]])
        Mnu = np.sqrt([[m1, 0, 0], [0, m2, 0], [0,0,m3]])

    elif H ==2:
        #IH case           
        th12, th13, th23  = (np.pi/180.0)*np.array([33.62, 8.58, 48.1])
        msolIH, matmIH = np.sqrt([7.40e-5*1e-18, 2.465e-3*1e-18])
        m1IH, m2IH, m3IH = np.sqrt(matmIH**2-msolIH**2), matmIH, 0.0  #IH
        # m2, m3 = m1IH, m2IH  # we want to use the same indices of masses for both cases

        Omega = np.matrix([[cw, sw],[-xi*sw, xi*cw],[0,0]])
        Mnu = np.sqrt([[m1IH, 0, 0], [0, m2IH, 0], [0,0,m3IH]])
        
    else:
        print('H should be 1 or 2 for NH or IH')
        

    s12, s13, s23 = np.sin([th12, th13, th23 ])
    c12, c13, c23 = np.cos([th12, th13, th23 ])
        # PMNS:
    U1 = np.matrix([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
    U2 = np.matrix([[c13, 0, s13*np.exp(-1.j*delta)], [0, 1, 0], [-s13*np.exp(1.j*delta), 0, c13]])
    U3 = np.matrix([[c12, s12, 0], [-s12, c12, 0], [0 ,0 , 1]])
    P = np.matrix([[1, 0, 0], [0, np.exp(1.j*eta), 0], [0, 0, 1]])

    U = U1*U2*U3*P # PMNS matrix


    return np.array(1.0j/v0 * U*Mnu*Omega*MN)

def thetaIa(M, dM, Imw, Rew, delta, eta,  H = 1, xi = 1):
    '''returns two arrays: Theta_2a and Theta_3a'''
    v0 = 174.1
    M2, M3 = M-dM, M+dM
    F = FM(M, dM, Imw, Rew, delta, eta, H, xi)
    Theta2a = v0/M2*F[:,0]
    Theta3a = v0/M3*F[:,1]
    return (Theta2a, Theta3a)

def theta_Ia_Sq(M, dM, Imw, Rew, delta, eta,  H = 1, xi = 1):
    '''returns two arrays: Theta_2a and Theta_3a'''
    th2, th3 = thetaIa(M, dM, Imw, Rew, delta, eta, H, xi)
    return (abs(th2)**2, abs(th3)**2)



# For cross check only
def Ua2(M, Xw, delta, eta, a, H = 1):
    '''returns U^2_a for given parameters.
    this realisation assumes that delta M / M = 0 so 
    there is no dependence on Re w.
    '''
    if H == 1:
        #NH case  
        th12, th13, th23  = (np.pi/180.0)*np.array([33.62, 8.54, 47.2])
        msol, matm = np.sqrt([7.40e-5*1e-18, 2.494e-3*1e-18])
        m1, m2, m3 = 0.0, msol, matm  #NH
    elif H ==2:
        #IH case           
        th12, th13, th23  = (np.pi/180.0)*np.array([33.62, 8.58, 48.1])
        msolIH, matmIH = np.sqrt([7.40e-5*1e-18, 2.465e-3*1e-18])
        m1IH, m2IH, m3IH = np.sqrt(matmIH**2-msolIH**2), matmIH, 0.0  #IH
        m2, m3 = m1IH, m2IH  # we want to use the same indices of masses for both cases
        
    else:
        print('H should be 1 or 2 for NH or IH')
        

    s12, s13, s23 = np.sin([th12, th13, th23 ])
    c12, c13, c23 = np.cos([th12, th13, th23 ])
        # PMNS:
    U1 = np.matrix([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
    U2 = np.matrix([[c13, 0, s13*np.exp(-1.j*delta)], [0, 1, 0], [-s13*np.exp(1.j*delta), 0, c13]])
    U3 = np.matrix([[c12, s12, 0], [-s12, c12, 0], [0 ,0 , 1]])
    P = np.matrix([[1, 0, 0], [0, np.exp(1.j*eta), 0], [0, 0, 1]])

    U = U1*U2*U3*P
    
    def Cpl(alpha):
        if H == 1:
            i2, i3 = 1, 2   #  indices start from 0, so 1, 2 corresponds to 2, 3 in SE note
        elif H == 2:
            i2, i3 = 0, 1  #  indices for IH (1 ,2 in SE note)
        a = alpha -1
        res = m2*np.abs(U[a, i2])**2 + m3*np.abs(U[a, i3])**2
        im = 2*np.sqrt(m2*m3)*np.imag(U[a,i2]* np.conj(U[a,i3]) )
        return res - im

    def Cmi(alpha):
        if H == 1:
            i2, i3 = 1, 2   #  indices start from 0, so 1, 2 corresponds to 2, 3 in SE note
        elif H == 2:
            i2, i3 = 0, 1  #  indices for IH (1 ,2 in SE note)
        a = alpha -1
        res = m2*np.abs(U[a, i2])**2 + m3*np.abs(U[a, i3])**2
        im = 2*np.sqrt(m2*m3)*np.imag(U[a,i2]* np.conj(U[a,i3]) )
        return res + im
    
    return (Cpl(a)*Xw**2 + Cmi(a)*Xw**(-2))/(2*M)

def U2list(M, Xw, delta, eta,  H = 1):
    Ue2 = Ua2(M, Xw, delta, eta, 1, H)
    Umu2 = Ua2(M, Xw, delta, eta, 2, H)
    Utau2 = Ua2(M, Xw, delta, eta, 3, H)
    return [Ue2, Umu2, Utau2]






if __name__ == '__main__':

    M, dM =  1, 1e-5
    Imw, Rew = 0.01, pi/2
    Xw = exp(Imw)
    delta, eta = pi/3, pi/5
    # print(FM(M, dM, Imw, Rew, delta, eta))

    # cross check
    # Fa = FM(M, dM, Imw, Rew, delta, eta)
    # ind = 0
    # print( (174.1/M)**2 *(abs(Fa[ind,0])**2 + abs(Fa[ind,1])**2))
    # ind = 1
    # print( (174.1/M)**2 *(abs(Fa[ind,0])**2 + abs(Fa[ind,1])**2))
    # ind = 2
    # print( (174.1/M)**2 *(abs(Fa[ind,0])**2 + abs(Fa[ind,1])**2))

    # print( U2list(M, Xw, delta, eta) )
    # print(' ')

    Fa = FM(M, dM, Imw, Rew, delta, eta)
    Th2, Th3 = theta_Ia_Sq(M, dM, Imw, Rew, delta, eta)
    print('FM:', Fa)
    # print(Fa[:,0])
    # print(abs(Th[0])**2)

    print(Th2+Th3)
    print( U2list(M, Xw, delta, eta) )






