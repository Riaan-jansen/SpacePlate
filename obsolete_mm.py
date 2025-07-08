'''
modal matching code from Murray thesis
'''

# result - colourmap frequency-depth+colour=transmission

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad  # obsolete

# hardcoded variables as defined in the paper
d = 8  # pitch (mm)
a = np.sqrt(np.pi * (2.4/2)**2)  # so circle and square have same area
rho = 1.225E-9  # air density (E-9kg/mmc)
rho_prime = rho
c = 295000  # (mm/s) STANDARD IS 343m/s but end corrections reduce
h1 = 60  # max pipe depth

N = 200  # to keep arrays same size
f1 = 5000; f2 = 40000  # (Hz)
frequency = np.linspace(f1, f2, int(N/25))  # for plotting
# k should be in (1/mm)
kx = 0; ky = 0  # changing these makes a big difference!
kz = np.linspace(f1*2*np.pi/c, f2*2*np.pi/c, N)
m = [0, 0]
# k = [kx, ky, kz]


def S(k, m):
    '''sum term'''
    k0_prime = np.sqrt(kx**2 + ky**2 + k**2)  # this is not necessarily the case
    s1 = (k0_prime * Q(k, m,  True) * Q(k, m,  False)) / (k * d**2)
    return s1  # sum(s1)


def Q(k, modes, positive):
    '''wavenumber(3), modes, positive or negative (bool). real and imaginary
    parts. numerically integrated, splitting e^ix into sin and cos.'''
    def Q_re(y, x, mode):
        phase = ((kx + 2*np.pi*mode[0]/d) * x + (ky + 2*np.pi*mode[1]/d) * y)
        if positive:
            sign = 1 
        else:
            sign = -1
        return np.cos(sign * phase)

    def Q_im(y, x, mode):
        phase = ((kx + 2*np.pi*mode[0]/d)*x + (ky + 2*np.pi*mode[1]/d) * y)
        if positive:
            sign = 1 
        else:
            sign = -1
        return np.sin(sign * phase)

    result_re = 0; result_im = 0

    # summing over all possible modes
    for mode in modes:
        if mode == 0:
            mode = [0,0]

        # integrates over x from 0 to a, and y from 0 to a
        q_re, _ = dblquad(Q_re, 0, a, lambda x: 0, lambda x: a, args=([mode]))
        q_im, _ = dblquad(Q_im, 0, a, lambda x: 0, lambda x: a, args=([mode]))

        result_re = result_re + q_re
        result_im = result_im + q_im

    return result_re + 1j * result_im


def Q_old(k, m, positive):
    '''taking the result of integration and evaluating that instead
    seems to give same values, only div 0 error when kx or ky = 0.'''
    if positive:
        sign = 1
    else:
        sign = -1
    q1 = (np.exp(sign*1j*a*(kx + (2*np.pi*m[0]/d))) - 1) / (sign*1j*(kx + (2*np.pi*m[0]/d)))
    q2 = (np.exp(sign*1j*a*(ky + (2*np.pi*m[1]/d))) - 1) / (sign*1j*(ky + (2*np.pi*m[1]/d)))
    q = q1 * q2
    return q


def T_old(k, m, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth.'''
    # k0 = sqrt(kx^2 + ky^2 + kz^2)
    k0 = np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k0  # change if considering losses

    T_numer = 4*(a**2) * np.exp(1j*k0*h) * rho * rho_prime *        \
                Q(k, m, False) * Q(k, [0, 0], True)

    # ORIGINAL - untampered replication of the equation from the paper
    # T1 = (-1 + np.exp(2j*k0*h)) * (S(k, m)**2) * (rho**2)
    # T2 = - ((2*a**2) * (1 + np.exp(2j*k0_prime*h)) * S(k, m) * rho * rho_prime)
    # T3 = ((a**4) * (-1 + np.exp(2j*k0_prime*h) * rho_prime**2) * k)
    # T_denom = (d**2) * (T1 + T2 + T3)

    # for tinkering
    T1 = (-1 + np.exp(2j*k0*h)) * (S(k, m)**2) * (rho**2)
    T2 = - ((2*a**2) * (1 + np.exp(2j*k0_prime*h)) * S(k, m) * rho * rho_prime)
    T3 = ((a**4) * (-1 + np.exp(2j*k0_prime*h)) * rho_prime**2 * k)
    T_denom = (d**2) * (T1 + T2 + T3)
    
    t = -T_numer / T_denom  # -T_numer / T_denom in the paper
    return t


def T(k, m, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''
    k0 = np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k0  # change if considering losses

    # defining terms
    e = np.exp(1j * k0_prime * h)
    alpha = (rho / rho_prime) * S(k, m)
    Q0 = Q(k, [0,0], True)  # positive, zero mode Q

    # A1 and A2 definitions by hand - these were obviously wrong :(
    # A1_numer = -(2*Q(k, [0,0], True) * (alpha + a**2) * e**(-2))
    # A1_denom = ((-alpha - a**2) * (alpha + a**2) + (alpha - a**2)**2)
    # A1 = A1_numer / A1_denom

    # A2_numer = -(2*Q(k, [0,0], True))
    # A2_denom = ((((-alpha - a**2) * (alpha + a**2)) / (alpha - a**2)) +  \
    # (alpha - a**2))
    # A2 = A2_numer / A2_denom

    # ================== VALUES FROM SIM_EQN_SOLVER.PY ==================
    A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 -   \
            2*a**2*alpha + alpha**2*e**2 - alpha**2)
    
    A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 -              \
            2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)

    # some prefactor of terms
    F_numer = rho * k0_prime * Q(k, m, False)
    F_denom = rho_prime * k * d**2

    t = (A1*e - A2*e**(-1)) * (F_numer / F_denom)
    return t


def colourmap(m_list, kz):
    '''takes list of modes (tuples) and plots Transmission spectra for each.
    and list kz (WHICH IS NOT YET A FUNCTION OF M ITSELF).
    '''
    # 2D grid of data
    kz = kz  # will probably have to come back to this for changing m
    h = np.linspace(1, h1, N)

    # convert to a form that colourmap works with (grid of points)
    H, K = np.meshgrid(h, kz)

    # setting up for looped plotting of subplots
    n = len(m_list)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
   
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # Make it easy to index

    for i, m in enumerate(m_list):
        # Z is SOME FUNCTION of X and Y AFTER they are converted to meshgrid
        Z = T(K, m, H)

        # for plotting all data (incl im) and normalisation
        Z = np.abs(Z)
        Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

        im = axes[i].imshow(Z_norm, extent=[h.min(), h.max(), kz.min(), kz.max()],
                            origin='lower', cmap='jet', aspect='auto')
        axes[i].set_title(f'Transmission for m=[{m[0]},{m[1]}]')
        axes[i].set_xlabel('Maximum Pipe Depth (mm)')
        axes[i].set_ylabel('Frequency (Hz)')
        # converts back to frequency (location on y, str label)
        axes[i].set_yticks(frequency*2*np.pi/c, ['%d' % val for val in frequency])
        fig.colorbar(im, ax=axes[i])

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot1D(k, m, h):
    ''''''
    # get transmission values
    Z = T(k, m, h)
    Z = np.abs(Z)
    Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    # plot against wavenumber
    plt.figure(figsize=(5,6))
    plt.plot(k, Z_norm, color='r')
    # plt.yscale('log')
    plt.title(f'Transmission for m = [{m[0]},{m[1]}] @ h = {h} mm')
    plt.xlabel('Frequency (Hz)')
    plt.xticks(frequency*2*np.pi/c, ['%d' % val for val in frequency])
    plt.ylabel('Transmission #')
    
    plt.grid()
    plt.show()


size = 3
# m_list = [(0,x) for x in range(size)] + [(x,0) for x in range(size)]
m_list = [[x,x] for x in range(size)]
print(m_list)
# colourmap(m_list, kz)

plot1D(kz, m=m_list, h=12)


# these variables appear in the appendix - think they are offset of centre
# of one fishnet hole to the other

h = a; g = a

def KZ(k, m):
    arg = k**2 - (kx + 2*m[0]*np.pi/d)**2 - (ky + 2*m[1]*np.pi/d)**2
    return np.sqrt(arg.astype(complex))


def Q_prime(k, mode, positive):
    '''completely unnecessary but its nice to have it called "Q_prime"'''
    if positive:
        sign = 1 
    else:
        sign = -1
    def Q_re(y, x, mode):
        phase = ((kx + 2*np.pi*mode[0]/d) * x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.cos(sign * phase)

    def Q_im(y, x, mode):
        phase = ((kx + 2*np.pi*mode[0]/d)*x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.sin(sign * phase)

    # integrates over x from 0 to a, and y from 0 to a
    q_re, _ = dblquad(Q_re, g, g+a, lambda x: h, lambda x: h+a, args=([mode]))
    q_im, _ = dblquad(Q_im, g, g+a, lambda x: h, lambda x: h+a, args=([mode]))

    return q_re + 1j * q_im


def T_DF(k, modes, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''
    # defining terms
    Q0 = Q(k, [0,0], True)  # positive, zero mode Q
    k0 = k # np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k0  # change if considering losses

    hg = 0.94
    h1 = h
    h2 = h + hg
    h3 = 2*h + hg

    def S1(k, mode):
        s1 = ( k0 * Q(k, mode, True) * Q_prime(k, mode, True) ) / (KZ(k, mode)\
             * d**2)
        return s1

    def S2(k, mode):
        s2 = 1j * k0 * (1/np.tan(KZ(k, mode)*h2)) * ( (Q(k, mode, True) *               \
                Q_prime(k, mode, True)) / (KZ(k, mode) * d**2) )
        return s2

    def S3(k, mode):
        s3 = 1j * k0 * (1/np.sin(KZ(k, mode)*h2)) * ( (Q(k, mode, True) * Q_prime(k, mode, \
            False)) / (KZ(k, mode) * d**2) )
        return s3

    def S4(k, mode):
        s4 = 1j * k0 * (1/np.sin(KZ(k, mode)*h2)) * ( (Q(k, mode, False) * Q_prime(k, mode, \
            True)) / (KZ(k, mode) * d**2) )
        return s4

    def S5(k, mode):
        s5 = 1j * k0 * (1/np.tan(KZ(k, mode)*h2)) * ( (Q(k, mode, False) * Q_prime(k, mode, \
            False)) / (KZ(k, mode) * d**2) )
        return s5

    def S6(k, mode):
        s6 = (k0 * Q(k, mode, False) * Q_prime(k, mode, False)) / (KZ(k, mode) * d**2)
        return s6

    def D(k, m):
        k0 = k
        s1 = S1(k, m); s2 = S2(k, m)
        s3 = S3(k, m); s4 = S4(k, m)
        s5 = S5(k, m); s6 = S6(k, m)
        
        D1 = a**2 * (s3*s4 - (s1 + s2)*(s5 + s6)) * np.cos(k0*h3)
        D2 = 1j * ((-s3*s4*s6 + (s1 + s2)*(a**4 + s5*s6))) * np.sin(k0*h3)
        
        prefactor1 = a**2 * np.cos(k0*h1)
        
        D3 = 1j * a**2 * (a**4 * (s5 + s6) + s1 * (-s3*s4 + s2*(s5 + s6))) * \
            np.cos(k0*h3)
        D4 = (a**8 + s1*(-s3*s4 + s2*s5)*s6 + a**4 * (s1*s2 + s5*s6)) *      \
            np.sin(k0*h3)
        
        prefactor2 = np.sin(k0*h1)

        d = (prefactor1 * (D1 + D2)) + (prefactor2 * (D3 + D4))
        return d

    # looping over n modes and summing the resultant t.
    T = 0
    for m in modes:
        t_numer = -(2*a**4) * S4(k,m) * Q0 * k0 * Q_prime(k,m,False)

        t_denom = D(k, m) * KZ(k, m) * d**2

        t = t_numer / t_denom 
        # t_norm = (t - np.min(t)) / (np.max(t) - np.min(t))

        T = T + t
    return T