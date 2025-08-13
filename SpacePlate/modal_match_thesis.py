'''
modal matching code from Murray thesis. Just for comparing theory to theory in
Figures 5.3-5.8.
Single and double fishnet cases.\n
- T() is single\n
- T_DF() is double\n
- colourmap() and plot1D() take DF=bool
'''

# result - colourmap frequency-depth+colour=transmission

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.special import jv

# hardcoded variables as defined in the paper
d = 8  # pitch (mm)
a = np.sqrt(np.pi * (2.4/2)**2)  # so circle and square have same area
rho = 1.225E-9  # air density (E-9kg/mmc)
rho_prime = rho
c = 343000  # (mm/s) STANDARD IS 343m/s
# h1 = 30  # max pipe depth
hg = 0.47

N = 200  # to keep arrays same size
f1 = 5000; f2 = 40000  # (Hz)
frequency = np.linspace(f1, f2, 8)  # for plotting
kx = 0; ky = 0  # k should be in (1/mm)
k = 2*np.pi*np.linspace(f1, f2, N)/c


def KZ(k, kx, m):
    '''input: k, kx, m. output: kz: array(complex).'''
    ky = 0
    m1, m2 = m
    arg = k**2 - (kx + 2*m1*np.pi/d)**2 - (ky + 2*m2*np.pi/d)**2
    arg = arg.astype(complex)
    return np.sqrt(arg)


def S(k, m):
    '''sum term'''
    # k0_prime = np.sqrt(kx**2 + ky**2 + k**2)  # this is not necessarily the case
    k0_prime = k
    kx = 0
    s1 = (k * Q(k, m, +1) * Q(k, m, -1)) / (KZ(k, kx, m) * d**2)
    return s1  # sum(s1)


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


def Q_bessel(kx, mode, sign):
    '''input: kx is some array, could equally be kx,ky tuple for changing both.
    mode is list of tuple pairs, sign is +/-1.
    output: Q: array.
    This is the analytical result of the surface integral of the im exponential
    where a is the upper limit of the integral and j1 is 1st kind Bessel func.'''
    ky = 0
    m1, m2 = mode
    alpha = kx + 2 * np.pi * m1 / d
    beta = ky + 2 * np.pi * m2 / d
    q = np.sqrt(alpha**2 + beta**2)
    # sign prefactor is not meant to be here (outside of j1)
    # removing sign / adding a sign* prefactor outside j1 has the same effect
    Q = np.where(q == 0, np.pi * a**2, sign * 2*np.pi * a * jv(1, sign * a * q) / q)
    return Q


def T(k, modes, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''
    T = 0
    kx = 0
    k0_prime = k

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
        F_denom = rho_prime * KZ(k, kx, m) * d**2

        t = (A1*e - A2*(1/e)) * (F_numer / F_denom)
        # print(f"Mode {m}: t = {t}")
        T = T + np.abs(t)
    return T


def T3(k, modes, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''
    T = 0
    kx = 0
    k0_prime = k
    e = np.exp(1j * k0_prime * h)
    Q0 = Q(k, (0,0), +1)  # positive, zero mode Q

    for m in modes:

        numer = -4*a**2 *e*rho*rho_prime*Q(k, m, -1)*Q(k, (0,0), +1)
        d1 = (-1+e**2)*(S(k, m))**2 *rho**2
        d2 = -2*a**2 *(1+e**2)*S(k,m)*rho*rho_prime
        d3 = a**4 * (-1 + e**2) *rho_prime**2 *KZ(k, kx, m)
        denom = d**2 * (d1 + d2 + d3)
        
        t = numer / denom
        T = T + t
    return T


def T2(k, modes, h):
    '''PAPER NOT THESIS T. function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''

    kx = 0
    a = 1.2
    e = np.exp(2j * k * h)
    e_inv = np.exp(-2j * k * h)

    S1 = np.zeros_like(k, dtype=complex)
    # looping over n modes and summing the resultant t.
    for m in modes:
        # defining terms

        q_plus = Q_DF(kx, m, +1)
        q_minus = Q_DF(kx, m, -1)
        kz = KZ(k, kx, m)
        S1 += k * q_minus * q_plus / (d**2 * kz)  # IS THIS THE RIGHT DEFINITION

    s1 = S1

    # prefactors for mode_00
    q_plus_00 = Q_DF(kx, (0,0), +1)
    q_minus_00 = Q_DF(kx, (0,0), -1)
    kz_00 = KZ(k, kx, (0,0))
    prefactor = k / (kz_00*d**2)
    polar = (np.pi*a**2 + s1)**2 / (np.pi*a**2)
    denom = e_inv*polar - e*polar

    T = prefactor * 4 * q_plus_00 * q_minus_00 / denom

    return T


def T4(k, modes, h):
    '''function to calculate transmission coeffs given k and m. where k is kz,
    m is integer mode in x and y, and h is pipe depth. revisited eqn for T
    again by hand solving for A1 A2.'''
    kx = 0
    k0_prime = k
    e = np.exp(2j * k0_prime * h)
    Q0 = Q(k, (0,0), +1)  # positive, zero mode Q

    s1 = np.zeros_like(k, dtype=complex)
    F_numer = np.zeros_like(k, dtype=complex)
    F_denom = np.zeros_like(k, dtype=complex)

    # looping over n modes and summing the resultant t.
    for m in modes:
        # defining terms
        s1 = s1 + S(k, m)
        # some prefactor of terms
        F_numer = F_numer + (rho * k0_prime * Q(k, m, -1))
        F_denom = F_denom + (rho_prime * KZ(k, kx, m) * d**2)

    alpha = (rho / rho_prime) * s1

    # ================== VALUES FROM SIM_EQN_SOLVER.PY =================
    # A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 -   \
    #         2*a**2*alpha + alpha**2*e**2 - alpha**2)
    
    # A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 -           \
    #         2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    
    A1 = (-2*Q0*a**2 - 2*Q0*alpha) / (a**4*e - a**4 - 2*a**2*alpha*e -   \
            2*a**2*alpha + alpha**2*e - alpha**2)
    
    A2 = (2*Q0*a**2*e - 2*Q0*alpha*e)/(a**4*e - a**4 -           \
            2*a**2*alpha*e - 2*a**2*alpha + alpha**2*e - alpha**2)

    e = np.exp(1j*h*k)

    # A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    # A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    T = (A1*e - A2*(1/e)) * (F_numer / F_denom)

    return T

# =======================================================================
# ============================ Double Fishnet ===========================

def Q_DF(kx, mode, sign):
    '''to match the definition of Q in the paper not thesis.
    kx has to be scalar here.'''
    r0 = 1.2  # mm

    def Q_re(r, theta, mode, kx):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        phase = ((kx + 2*np.pi*mode[0]/d) * x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.cos(sign * phase) * r

    def Q_im(r, theta, mode, kx):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        phase = ((kx + 2*np.pi*mode[0]/d)*x + (ky + 2*np.pi*mode[1]/d) * y)
        return np.sin(sign * phase) * r

    # integrates over x from 0 to a, and y from 0 to a
    q_re, _ = dblquad(Q_re, 0, 2*np.pi, lambda x: 0, lambda x: r0, args=(mode, kx))
    q_im, _ = dblquad(Q_im, 0, 2*np.pi, lambda x: 0, lambda x: r0, args=(mode, kx))

    return q_re + 1j * q_im


def T_DF(k, modes, h, hg=0.94):
    '''Double Fishnet. equation for 00 order T from paper on holey structures'''
    k0 = k  # np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k0  # change if considering losses
    kx = 0

    a = 1.2  # circle radius

    hg = 0.94  # 0.94 or 0.47 mm
    # h1 = h; h2 = h + hg; h3 = 2*h + hg
    def QQ(kx, m):
        qq = Q_DF(kx, m, -1) * Q_DF(kx, m, +1)
        return qq

    def S1(k, kx, m):
        s1 = k * QQ(kx, m) / (d**2 * KZ(k, kx, m))
        return s1
    
    def S2(k, kx, m):
        s2 = 1j * (1/np.tan(KZ(k, kx, m)*hg)) * k * QQ(kx, m) / (d**2 * KZ(k, kx, m))
        return s2
    
    def S3(k, kx, m):
        s3 = 1j * (1/np.sin(KZ(k, kx, m)*hg)) * k * QQ(kx, m) / (d**2 * KZ(k, kx, m))
        return s3

    def D(k, modes):
        '''modes'''
        s1 = np.zeros_like(k)
        s2 = np.zeros_like(k)
        s3 = np.zeros_like(k)
        for m in modes:

            s1 = s1 + S1(k, kx, m)
            s2 = s2 + S2(k, kx, m)
            s3 = s3 + S3(k, kx, m)

        D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi * a**2 +\
            s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
        
        D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi * a**2 - \
            s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
    
        D3 = -2 * (np.pi**2 * a**4 - s1**2) * (np.pi**2 * a**4 -       \
            s2**2 + s3**2) / (2*np.pi**2 * a**4 * s3)
        
        d = D1 + D2 + D3
        return d

    T0 = k * 4 * QQ(kx, (0,0)) /  \
        (d**2 * KZ(k, kx, (0, 0)) * D(k, modes))

    return T0


def T_SF(k, modes, h):
    '''Checking for convergence in solutions for DF and SF cases. equation for 00 order T from paper on holey structures'''
    k0 = k  # np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
    k0_prime = k0  # change if considering losses
    kx = 0

    a = 1.2  # circle radius

    hg = 0.001  # 0.94 or 0.47 mm
    # h1 = h; h2 = h + hg; h3 = 2*h + hg
    def QQ(kx, m):
        qq = Q_DF(kx, m, -1) * Q_DF(kx, m, +1)
        return qq

    def S1(k, kx, m):
        s1 = k * QQ(kx, m) / (d**2 * KZ(k, kx, m))
        return s1
    
    def S2(k, kx, m):
        s2 = 1j * (1/np.tan(KZ(k, kx, m)*hg)) * k * QQ(kx, m) / (d**2 * KZ(k, kx, m))
        return s2
    
    def S3(k, kx, m):
        s3 = 1j * (1/np.sin(KZ(k, kx, m)*hg)) * k * QQ(kx, m) / (d**2 * KZ(k, kx, m))
        return s3

    def D(k, modes):
        '''modes'''
        s1 = np.zeros_like(k)
        s2 = np.zeros_like(k)
        s3 = np.zeros_like(k)
        for m in modes:

            s1 = s1 + S1(k, kx, m)
            # s2 = s2 + S2(k, kx, m)
            # s3 = s3 + S3(k, kx, m)
            s2 = np.ones_like(k)*0.1
            s3 = np.ones_like(k)*0.1

        D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi * a**2 +\
            s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
        
        D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi * a**2 - \
            s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
    
        D3 = -2 * (np.pi**2 * a**4 - s1**2) * (np.pi**2 * a**4 -       \
            s2**2 + s3**2) / (2*np.pi**2 * a**4 * s3)
        
        d = D1 + D2 + D3
        return d

    Q_00 = QQ(kx, (0,0))

    T0 = k * 4 * Q_00 /  \
        (d**2 * KZ(k, kx, (0, 0)) * D(k, modes))

    return T0
# =======================================================================
# ============================ Plotting ===========================

def colourmap(m_list, k, DF=False):
    '''takes list of modes (tuples) and plots Transmission spectra for each.
    and list kz (WHICH IS NOT YET A FUNCTION OF M ITSELF).
    '''
    # 2D grid of data
    # convert to a form that colourmap works with (grid of points)

    if DF is True:
        h1 = 30
        h = np.linspace(1, h1, N)
        H, K = np.meshgrid(h, k)
        # Z is SOME FUNCTION of X and Y AFTER they are converted to meshgrid
        Z = T_DF(K, m_list, H)
        # for plotting all data (incl im) and normalisation
        Z = np.abs(Z)
        Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        plt.imshow(Z_norm, extent=[h.min(), h.max(), k.min(), k.max()],
                    origin='lower', cmap='plasma', aspect='auto')
    else:
        h1 = 60
        h = np.linspace(1, h1, N)
        H, K = np.meshgrid(h, k)
        Z = T_SF(K, m_list, H)
        Z = np.abs(Z)
        Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        plt.imshow(Z_norm, extent=[h.min(), h.max(), k.min(), k.max()],
                    origin='lower', cmap='jet', aspect='auto')

    plt.colorbar(label='Transmission')
    plt.title(f'Transmission Spectra for $h_g$ = {hg}mm')
    plt.xlabel('Plate Depth (mm)')
    plt.ylabel('Frequency (kHz)')
    plt.ylim(bottom=10000*2*np.pi/c)
    # converts back to frequency (location on y, str label)
    plt.yticks(frequency*2*np.pi/c, ['%d' % (val/1000) for val in frequency])

    plt.show()


def plot1D(k, m, h, DF=False):
    '''plots Transmission vs frequency for fixed h.'''
    # get transmission values
    if DF is True:
        Z = T_DF(k, m, h)
    else:
        Z = T4(k, m, h)
    Z = np.abs(Z)**2

    # Z_old = T3(k, m, h)
    # Z_old = np.abs(Z_old)

    # plot against wavenumber
    plt.figure(figsize=(5,4))
    plt.plot(k, Z, color='r')

    plt.yscale('log', base=10)
    plt.title(f'Transmission @ h = {h} mm')
    plt.xlabel('Frequency (Hz)')
    plt.xticks(frequency*2*np.pi/c, ['%d' % val for val in frequency])
    plt.ylabel('Transmission #')
    
    plt.grid()
    plt.show()


size = 2
m_list = [(m1, m2) for m1 in range(-size, size+1) for m2 in range(-size, size+1)]
m_list = list(dict.fromkeys(m_list))  # dict does not have duplicate entries by definition
# m_list = [(0,x) for x in range(size)]

# m_list = [(0,0)]
# ============================================================================
# ============================ calling ==================================
# # print(m_list)
# colourmap(m_list, k, DF=True)
# # colourmap(m_list, k, DF=False)
# plot1D(k, m_list, h=12, DF=True)
# plot1D(k, m_list, h=12, DF=False)

# bragg condition - calculating allowed orders
# n = 2*d*f2/c
# print(n)  # n = 2
