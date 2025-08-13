'''
TO PLOT DISPERSION RELATION: dispersion_plot()\n
TO PLOT TRANSMISSION AMPITUDE: amplitude_plot()\n
Calculates the dispersion relation k0 against kx against transmission
coefficient. Sweeping over angle. PAPER refers to "Low acoustic transmittance 
through a holey structure (2012) Bell".\n
rewritten functions for calculating T to loop over kx as well.\n
- T_dispersion() calculates the coefficients given arrays KX K0.\n
- dispersion_plot() plots k0 - kx mapping T with scatter points.\n
Old values to match Murray thesis: d=8, a=1.2, hg=0.47/0.94
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # bessel function vth kind jv(v, arg)

from scipy.integrate import dblquad

################################ variables ####################################
d = 6.7  # pitch (mm)
# a = np.sqrt(np.pi * (0.0024/2)**2)  # so circle and square have same area
a = 1.3
rho = 1.18E-9  # 1.225E-9  # air density (E-9kg/mmc)
rho_prime = rho
c = 344820  # (mm/s) STANDARD IS 343m/s
h = 1.5  # max pipe depth
hg = 19.6

# size of arrays
N = 400
# frequency range (Hz)
f1 = 10; f2 = 40000
fn = 5000
frequency = np.arange(f1, f2, fn)  # for plotting
ky = 0  # k should be in (1/mm)


################################ functions ####################################
def KZ(k, kx, m):
    '''input: k, kx, m. output: kz: array(complex).'''
    ky = 0
    m1, m2 = m
    arg = k**2 - (kx + 2*m1*np.pi/d)**2 - (ky + 2*m2*np.pi/d)**2
    arg = arg.astype(complex)
    return np.sqrt(arg)


def Q_bessel(kx, mode, sign):
    '''input: kx is some array, could equally be kx,ky tuple for changing both.
    mode is list of tuple pairs, sign is +/-1.\n
    output: Q: array.\n
    This is the analytical result of the surface integral of the im exponential
    where a is the upper limit of the integral and j1 is 1st kind Bessel func.'''
    ky = 0
    m1, m2 = mode
    alpha = kx + 2 * np.pi * m1 / d
    beta = ky + 2 * np.pi * m2 / d
    q = np.sqrt(alpha**2 + beta**2)
    # np.where(condition i.e. r=0, do if true, else do this)
    Q = np.where(q == 0, np.pi * a**2,  sign* 2*np.pi * a * jv(1, sign* a * q) / q)
    return Q


def Q_DF(kx, mode, sign):
    '''obsolete
    to match the definition of Q in the paper not thesis.
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


def T(kx, k, modes):
    '''equation for T from paper\n
    input: kx: array, k: array, modes: list of tuples.\n
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
        S1 += k * qq / (d**2 * kz)
        S2 += 1j * (1/np.tan(kz*hg)) * k * qq / (d**2 * kz)
        S3 += 1j * (1/np.sin(kz*hg)) * k * qq / (d**2 * kz)

    s1, s2, s3 = S1, S2, S3
    # original
    D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi*a**2 + s2)**2 - s3**2) / (2*np.pi**2 * (a**4) * s3)
    D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi*a**2 - s2)**2 - s3**2) / (2*np.pi**2 * (a**4) * s3)
    D3 = -2 * (np.pi**2 * (a**4) - s1**2) * (np.pi**2 * (a**4) - s2**2 + s3**2) / (2*np.pi**2 * (a**4) * s3)
    D = D1 + D2 + D3

    # prefactors for mode_00
    qq00 = QQ(kx, (0,0))
    kz_00 = KZ(k, kx, (0,0))

    # equation from paper
    t = k * 4 * qq00 / (d**2 * kz_00 * D)
    return t


def dispersion_plot():
    '''input: KX: array, k: array, modes: list of tuple.\n
    output: plots colourmap of KX-frequency with transmission coeffs as colour.
    '''
    # defines angle and frequency to sweep over
    max_angle = 89.9 # do NOT set to 90 - doesnt like it
    thetas = np.linspace(0, max_angle*np.pi/180, N)
    k0 = 2*np.pi/c * np.linspace(f1, f2, N)

    K0, THETA = np.meshgrid(k0, thetas)
    KX = K0 * np.sin(THETA)  # kx is now NxN

    # list of modes (not repeating) range [-3,3] for m1 and m2
    size = 3
    modes = [(m1, m2) for m1 in range(-size+1, size) for m2 in range(-size+1, size)]
    modes = list(dict.fromkeys(modes))

    Z = T(KX, K0, modes)
    Z = np.abs(Z)

    Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    # plt.imshow(Z_norm.T, extent=[KX.min(), KX.max(), K0.min(), K0.max()],
    #        origin='lower', cmap='plasma', aspect='auto')
    plt.pcolormesh(KX, K0, Z_norm, shading='auto', cmap='plasma')


    #plt.figure(figsize=(6,6))
    plt.colorbar(label='Transmission')
    plt.title(f'Spaceplate Dispersion Relation - Modal Matching')
    plt.xlabel('$k_x$ (1/mm)')
    plt.xlim(right=0.47)
    plt.ylabel('Frequency (kHz)')
    plt.yticks(frequency*2*np.pi/c, ['%d' % (val/1000) for val in frequency])
    plt.show()


def amplitude_plot():
    '''plots transmission at normal incidence for all frequencies according to
    modal matching theory'''
    # N needs to be bigger to cover frequencies where T -> 1
    N = 1800
    k0 = 2*np.pi/c * np.linspace(f1, f2, N)

    # list of modes (not repeating) range [-3,3] for m1 and m2
    size = 3
    modes= [(m1, m2) for m1 in range(-size+1, size) for m2 in range(-size+1, size)]
    modes = list(dict.fromkeys(modes))

    Z = T(0, k0, modes)  # kx = 0 == normal incidence
    Z = np.abs(Z)**2
    # normalised
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    plt.plot(np.linspace(f1, f2, N)/1000, Z)

    plt.title(f'Modal Matching - Amplitude of Transmission')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('$|T|^2$')
    plt.grid()
    plt.show()


def dispersion_map():
    '''Obsolete.
    plots using scatter points. accounts for uneven spacing in array
    with sin(t) dependency.'''
    return

    # defines angle and frequency to sweep over
    thetas = np.linspace(0, 89.9*np.pi/180, N)
    k0 = 2*np.pi/c * np.linspace(f1, f2, N)

    K0, THETA = np.meshgrid(k0, thetas)
    KX = K0 * np.sin(THETA)  # kx is now NxN

    # list of modes (not repeating) range [-3,3] for m1 and m2
    size = 3
    modes= [(m1, m2) for m1 in range(-size+1, size) for m2 in range(-size+1, size)]
    modes = list(dict.fromkeys(modes))

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

    plt.show()

################################# calling #####################################
# dispersion_plot()
# amplitude_plot()

