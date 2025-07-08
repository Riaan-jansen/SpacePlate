'''
modal matching code from Murray thesis
'''

# result - colourmap frequency-depth+colour=transmission

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# hardcoded variables as defined in the paper
d = 8  # pitch (mm)
a = np.sqrt(np.pi * (2.4/2)**2)  # so circle and square have same area
rho = 1.225E-9  # air density (E-9kg/mmc)
rho_prime = rho
c = 343000  # (mm/s) STANDARD IS 343m/s
h1 = 60  # max pipe depth

N = 200  # to keep arrays same size
f1 = 5000; f2 = 40000  # (Hz)
frequency = np.linspace(f1, f2, 8)  # for plotting
kx = 0; ky = 0  # k should be in (1/mm)
k = 2*np.pi/c * np.linspace(f1, f2, N)


def KZ(k, m):
    arg = k**2 - (kx + 2*m[0]*np.pi/d)**2 - (ky + 2*m[1]*np.pi/d)**2
    return np.sqrt(arg.astype(complex))


def Q(k, mode, sign):
    '''wavenumber(3), modes, positive or negative (bool). real and imaginary
    parts. numerically integrated, splitting e^ix into sin and cos.'''

    def Q_re(y, x, mode):
        phase = ((kx + 2*np.pi*mode[0]/d) * x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.cos(sign * phase)

    def Q_im(y, x, mode):
        phase = ((kx + 2*np.pi*mode[0]/d)*x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.sin(sign * phase)

    # integrates over x from 0 to a, and y from 0 to a
    q_re, _ = dblquad(Q_re, 0, a, lambda x: 0, lambda x: a, args=([mode]))
    q_im, _ = dblquad(Q_im, 0, a, lambda x: 0, lambda x: a, args=([mode]))

    return q_re + 1j * q_im


def S(k, m):
    '''sum term'''
    # k0_prime = np.sqrt(kx**2 + ky**2 + k**2)  # this is not necessarily the case
    k0_prime = k
    s1 = (k0_prime * Q(k, m, +1) * Q(k, m, -1)) / (KZ(k, m) * d**2)
    return s1  # sum(s1)


def T(k, modes, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''
    T = 0
    k0 = k  # k0 = np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k  # change if considering losses

    # looping over n modes and summing the resultant t.
    for m in modes:
        # defining terms
        e = np.exp(1j * k0_prime * h)
        alpha = (rho / rho_prime) * S(k, m)
        Q0 = Q(k, (0,0), +1)  # positive, zero mode Q

        # ================== VALUES FROM SIM_EQN_SOLVER.PY ==================
        A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 -   \
                2*a**2*alpha + alpha**2*e**2 - alpha**2)
        
        A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 -           \
                2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)

        # some prefactor of terms
        F_numer = rho * k0_prime * Q(k, m, -1)
        F_denom = rho_prime * KZ(k, m) * d**2

        t = (A1*e - A2*e**(-1)) * (F_numer / F_denom)
        # print(f"Mode {m}: t = {t}")
        T = T + t
    return T

# =======================================================================
# ============================ Double Fishnet ===========================

def Q_DF(kx, mode, sign):
    '''to match the definition of Q in the paper not thesis'''
    r0 = 1.2  # mm

    def Q_re(r, theta, mode):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        phase = ((kx + 2*np.pi*mode[0]/d) * x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.cos(sign * phase) * r

    def Q_im(r, theta, mode):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        phase = ((kx + 2*np.pi*mode[0]/d)*x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.sin(sign * phase) * r

    # integrates over x from 0 to a, and y from 0 to a
    q_re, _ = dblquad(Q_re, 0, 2*np.pi, lambda x: 0, lambda x: r0, args=([mode]))
    q_im, _ = dblquad(Q_im, 0, 2*np.pi, lambda x: 0, lambda x: r0, args=([mode]))

    return q_re + 1j * q_im


def T_DF2(k, modes, h):
    '''equation for 0 order T from paper on holey structures'''
    k0 = k  # np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k0  # change if considering losses

    a = 1.2  # circle radius

    hg = 0.47  # 0.94 mm
    h1 = h
    h2 = h + hg
    h3 = 2*h + hg

    def S1(k, modes):
        sum1 = 0
        for m in modes:
            s1 = k * Q_DF(kx, m, -1) * Q_DF(kx, m, +1) / (d**2 * KZ(k, m))
            sum1 = sum1 + s1
        return sum1
    
    def S2(k, modes):
        sum2 = 0
        for m in modes:
            s2 = 1j * (1/np.tan(KZ(k, m)*hg)) * k * Q_DF(kx, m, -1) * Q_DF(kx, m, +1)   \
                / (d**2 * KZ(k, m))
            sum2 = sum2 + s2
        return sum2
    
    def S3(k, modes):
        sum3 = 0
        for m in modes:
            s3 = 1j * (1/np.sin(KZ(k, m)*hg)) * k * Q_DF(kx, m, -1) * Q_DF(kx, m, +1)   \
                / (d**2 * KZ(k, m))
            sum3 = sum3 + s3
        return sum3

    def D(k, m):
        '''m = modes'''
        s1 = S1(k, m)
        s2 = S2(k, m)
        s3 = S3(k, m)

        D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi * a**2 +\
            s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
        
        D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi * a**2 - \
            s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
    
        D3 = -2 * (np.pi**2 * a**4 - s1**2) * (np.pi**2 * a**4 -       \
            s2**2 + s3**2) / (2*np.pi**2 * a**4 * s3)
        
        d = D1 + D2 + D3
        return d

    T0 = k * 4 * Q_DF(kx, (0, 0), +1) * Q_DF(kx, (0, 0), -1) /  \
        (d**2 * KZ(k, (0, 0)) * D(k, modes))

    return T0

# =======================================================================
# ============================ Plotting ===========================

def colourmap(m_list, k, h1, DF=False):
    '''takes list of modes (tuples) and plots Transmission spectra for each.
    and list kz (WHICH IS NOT YET A FUNCTION OF M ITSELF).
    '''
    # 2D grid of data
    # convert to a form that colourmap works with (grid of points)
    h = np.linspace(1, h1, N)
    H, K = np.meshgrid(h, k)

    if DF is True:
    # Z is SOME FUNCTION of X and Y AFTER they are converted to meshgrid
        Z = T_DF2(K, m_list, H)
        # for plotting all data (incl im) and normalisation
        Z = np.abs(Z)
        Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        plt.imshow(Z_norm, extent=[h.min(), 30, k.min(), k.max()],
                    origin='lower', cmap='jet', aspect='auto')
    else:
        Z = T(K, m_list, H)
        Z = np.abs(Z)
        Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        plt.imshow(Z_norm, extent=[h.min(), h.max(), k.min(), k.max()],
                    origin='lower', cmap='jet', aspect='auto')

    plt.colorbar(label='Transmission')
    plt.title(f'Transmission for m_max={max(m_list)}')
    plt.xlabel('Maximum Plate Depth (mm)')
    plt.ylabel('Frequency (kHz)')
    # converts back to frequency (location on y, str label)
    plt.yticks(frequency*2*np.pi/c, ['%d' % (val/1000) for val in frequency])

    plt.show()


def plot1D(k, m, h, DF=False):
    '''plots Transmission vs frequency for fixed h.'''
    # get transmission values
    if DF is True:
        Z = T_DF2(k, m, h)
    else:
        Z = T(k, m, h)
    Z = np.abs(Z)

    # plot against wavenumber
    plt.figure(figsize=(8,6))
    plt.plot(k, Z, color='r')
    plt.yscale('log', base=2)
    plt.title(f'Transmission @ h = {h} mm')
    plt.xlabel('Frequency (Hz)')
    plt.xticks(frequency*2*np.pi/c, ['%d' % val for val in frequency])
    plt.ylabel('Transmission #')
    
    plt.grid()
    plt.show()


size = 3
m_list = [(x,x) for x in range(size)] + [(0,x) for x in range(size)] + [(x,0) for x in range(size)]
# m_list = [(m1, m2) for m1 in range(-size, size+1) for m2 in range(-size, size+1)]
m_list = list(dict.fromkeys(m_list))  # dict does not have duplicate entries by definition
# m_list = [(0,x) for x in range(size)]

print(m_list)
colourmap(m_list, k, h1, DF=True)
colourmap([(0,0)], k, h1, DF=False)
# plot1D(k, m=m_list, h=12, DF=True)
# kxs = np.linspace(0, 400, 400)
# kxs = 0.35
# T_dspn(k, m_list, kxs)

# colourmap(m_list, k)
# plot1D(k, m=m_list, h=12)

# bragg condition - calculating allowed orders
# n = 2*d*f2/c
# print(n)  # n = 2
