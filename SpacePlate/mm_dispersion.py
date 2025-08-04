'''
Calculates the dispersion relation k0 against kx against transmission
coefficient. Sweeping over angle.
rewritten functions for calculating T to loop over kx as well.\n
- T_dispersion() calculates the coefficients given arrays KX K0.\n
- dispersion_plot() plots k0 - kx mapping T with scatter points.\n
Old values to match theory: d=8, a=1.2, hg=0.47/0.94
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, j0, j1  # bessel function 1st kind
import matplotlib.colors as mcolors
from scipy.integrate import dblquad

# hardcoded variables as defined in the paper
d = 6.75  # pitch (mm)
# a = np.sqrt(np.pi * (0.0024/2)**2)  # so circle and square have same area
a = 1.25
rho = 1.225E-9  # air density (E-9kg/mmc)
rho_prime = rho
c = 343000  # (mm/s) STANDARD IS 343m/s but end corrections reduce
h = 1.6  # max pipe depth
hg = 18.9

N = 400  # to keep arrays same size
f1 = 100; f2 = 40000  # (Hz)
frequency = np.linspace(f1, f2, 9)  # for plotting
ky = 0  # k should be in (1/mm)

# defines angle and frequency to sweep over
thetas = np.linspace(0, 89.9*np.pi/180, N)
k0 = 2*np.pi/c * np.linspace(f1, f2, N)

K0, THETA = np.meshgrid(k0, thetas)
KX = K0 * np.sin(THETA)  # kx is now NxN


def KZ(k, kx, m):
    '''input: k, kx, m. output: kz: array(complex).'''
    ky = 0
    m1, m2 = m
    arg = k**2 - (kx + 2*m1*np.pi/d)**2 - (ky + 2*m2*np.pi/d)**2
    arg = arg.astype(complex)
    return np.sqrt(arg)


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


def Q_DF(kx, mode, sign):
    '''to match the definition of Q in the paper not thesis.
    kx has to be scalar here.'''

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
    q_re, _ = dblquad(Q_re, 0, 2*np.pi, lambda x: 0, lambda x: a, args=(mode, kx))
    q_im, _ = dblquad(Q_im, 0, 2*np.pi, lambda x: 0, lambda x: a, args=(mode, kx))

    return q_re + 1j * q_im


def T_dispersion(kx, k, modes):
    '''input: kx: array, k: array, modes: list of tuples.
    output: T: array of Transmission coefficients.'''
    def QQ(kx, m):
        q_minus = Q_bessel(kx, m, -1)
        q_plus = Q_bessel(kx, m, +1)
        return q_minus * q_plus

    # S1, S2, S3 as arrays
    S1 = np.zeros_like(k, dtype=complex)
    S2 = np.zeros_like(k, dtype=complex)
    S3 = np.zeros_like(k, dtype=complex)

    # summing over all modes, for each element in S += each k element contribution
    for m in modes:
        kz = KZ(k, kx, m)
        qq = QQ(kx, m)
        # original
        # S1 += k * q_minus * q_plus / (d**2 * kz)
        # S2 += 1j * (1/np.tan(kz*hg)) * k * q_minus * q_plus / (d**2 * kz)
        # S3 += 1j * (1/np.sin(kz*hg)) * k * q_minus * q_plus / (d**2 * kz)
        S1 += k * qq / (d**2 * kz)
        S2 += 1j * (1/np.tan(kz*hg)) * k * qq / (d**2 * kz)
        S3 += 1j * (1/np.sin(kz*hg)) * k * qq / (d**2 * kz)

    s1, s2, s3 = S1, S2, S3
    # original
    #D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi*a**2 + s2)**2 - s3**2) / (2*np.pi**2 * (a**4) * s3)
    #D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi*a**2 - s2)**2 - s3**2) / (2*np.pi**2 * (a**4) * s3)
    #D3 = -2 * (np.pi**2 * (a**4) - s1**2) * (np.pi**2 * (a**4) - s2**2 + s3**2) / (2*np.pi**2 * (a**4) * s3)
    
    D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi*a**2 + s2)**2 - s3**2) / (2*np.pi**2 * (a**4) * s3)
    D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi*a**2 - s2)**2 - s3**2) / (2*np.pi**2 * (a**4) * s3)
    D3 = -2 * (np.pi**2 * (a**4) - s1**2) * (np.pi**2 * (a**4) - s2**2 + s3**2) / (2*np.pi**2 * (a**4) * s3)
    D = D1 + D2 + D3

    # prefactors for mode_00
    qq00 = QQ(kx, (0,0))
    kz_00 = KZ(k, kx, (0,0))

    # equation from paper
    T = k * 4 * qq00 / (d**2 * kz_00 * D)
    return T


def TSF_dispersion(kx, k, modes):
    '''T values single fishnet case.\nInput: kx, k, modes (all arrays).'''
    S1 = np.zeros_like(k, dtype=complex)
    F_numer = np.zeros_like(k, dtype=complex)
    F_denom = np.zeros_like(k, dtype=complex)

    Q0 = Q_bessel(kx, (0, 0), +1)
    e = np.exp(2j * k * h)

    # summing over all modes, for each element in S += each k element contribution
    for m in modes:
        kz = KZ(k, kx, m)
        q_minus = Q_bessel(kx, m, -1)
        q_plus = Q_bessel(kx, m, +1)

        S1 += k * q_minus * q_plus / (d**2 * kz)    

        # some prefactor of terms
        F_numer = F_numer + (rho * k * Q_bessel(k, m, -1))
        F_denom = F_denom + (rho_prime * KZ(k, kx, m) * d**2)

    s1 = S1
    f = F_numer/F_denom

    alpha = (rho / rho_prime) * s1

    # ================== VALUES FROM SIM_EQN_SOLVER.PY =================
    # A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 -   \
    #         2*a**2*alpha + alpha**2*e**2 - alpha**2)
    
    # A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 -           \
    #         2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    
    # A1 = (-2*Q0*a**2 - 2*Q0*alpha) / (a**4*e - a**4 - 2*a**2*alpha*e -   \
    #         2*a**2*alpha + alpha**2*e - alpha**2)
    
    # A2 = (2*Q0*a**2*e - 2*Q0*alpha*e)/(a**4*e - a**4 -           \
    #         2*a**2*alpha*e - 2*a**2*alpha + alpha**2*e - alpha**2)

    e = np.exp(1j*k*h)
    A1 = (-2*Q0*a**2 - 2*Q0*alpha)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)
    A2 = (2*Q0*a**2*e**2 - 2*Q0*alpha*e**2)/(a**4*e**2 - a**4 - 2*a**2*alpha*e**2 - 2*a**2*alpha + alpha**2*e**2 - alpha**2)

    T = (A1*e - A2*(1/e)) * f

    return T


def dispersion_map(KX, K0, modes, log=False):
    '''
    obsolete/needs work
    input: KX: array, k: array, modes: list of tuple, optional: log: bool.
    output: plots colourmap of KX-frequency with transmission coeffs as colour.'''

    Z = T_dispersion(KX, K0, modes)
    Z = np.abs(Z)

    if log is False:
        Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        # plt.imshow(Z_norm.T, extent=[KX.min(), KX.max(), K0.min(), K0.max()],
        #        origin='lower', cmap='plasma', aspect='auto')
        plt.pcolormesh(KX, K0, Z_norm, shading='auto', cmap='plasma')
    else:
        Z[Z == 0] = 1e-6  # some arbitrary floor for log plotting
        norm = mcolors.LogNorm(vmin=Z.min(), vmax=Z.max())
        plt.imshow(Z.T, extent=[KX.min(), KX.min(), K0.min(), K0.max()],
               origin='lower', cmap='plasma', aspect='auto', norm=norm)

    #plt.figure(figsize=(6,6))
    plt.colorbar(label='Transmission')
    plt.title(f'Spaceplate Dispersion Relation - Modal Matching')
    plt.xlabel('$k_x$ (1/mm)')
    plt.xlim(right=0.47)
    plt.ylabel('Frequency (kHz)')
    plt.yticks(frequency*2*np.pi/c, ['%d' % (val/1000) for val in frequency])
    plt.show()


def dispersion_plot(KX, K0, modes):
    '''plots using scatter points. accounts for uneven spacing in array
    with sin(t) dependency.'''

    Z = TSF_dispersion(KX, K0, modes)
    Z = np.abs(Z)
    Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    plt.scatter(KX, K0, s=1, c=Z_norm, cmap='jet')
    plt.colorbar(label='Transmission coeff')
    plt.title(f'Transmission for m_max={max([max(m) for m in modes])}')
    plt.xlabel('$k_x$ (1/mm)')
    plt.ylabel('Frequency (kHz)')

    plt.yticks(frequency*2*np.pi/c, ['%d' % (val/1000) for val in frequency])
    plt.xlim(left=0.0, right=0.4)
    plt.ylim(top=2*np.pi*f2/c, bottom=2*np.pi*f1/c)

    plt.tight_layout()
    plt.show()

size = 3
m_list = [(m1, m2) for m1 in range(-size+1, size) for m2 in range(-size+1, size)]
m_list = list(dict.fromkeys(m_list))
# m_list=[(0,0)]

#dispersion_map(KX, K0, m_list)
# dispersion_map(KX, K0, m_list, log=False)

# x = np.linspace(0, 1, N*2)
# y = np.where(x == 0, np.pi * a**2, 2*np.pi * a * jv(1, a * x) / x)
# plt.scatter(x, y, s=1)
# plt.title('y = np.where(x == 0, np.pi * a**2, 2*np.pi * a * jv(1, a * x) / x)')
# plt.show()
