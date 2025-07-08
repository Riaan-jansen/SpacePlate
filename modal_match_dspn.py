import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.special import j1  # bessel function 1st kind

# hardcoded variables as defined in the paper
d = 8  # pitch (mm)
# a = np.sqrt(np.pi * (2.4/2)**2)  # so circle and square have same area
a = 1.2
rho = 1.225E-9  # air density (E-9kg/mmc)
rho_prime = rho
c = 295000  # (mm/s) STANDARD IS 343m/s but end corrections reduce
h1 = 30  # max pipe depth

N = 100  # to keep arrays same size
f1 = 5000; f2 = 40000  # (Hz)
frequency = np.linspace(f1, f2, 8)  # for plotting
# k should be in (1/mm)
ky = 0  # changing these makes a big difference!
# k = np.linspace(f1*2*np.pi/c, f2*2*np.pi/c, N)

# def KZ(k, m, kx):
#     arg = k**2 - (kx + 2*m[0]*np.pi/d)**2 - (ky + 2*m[1]*np.pi/d)**2
#     arg = arg.astype(complex)
#     return np.sqrt(arg)

# def Q_DF(kx, mode, positive):
#     '''to match the definition of Q in the paper not thesis'''
#     r0 = 1.2  # mm

#     if positive:
#         sign = 1 
#     else:
#         sign = -1

#     def Q_re(r, theta, mode, kx):
#         x = r * np.cos(theta)
#         y = r * np.sin(theta)
#         phase = ((kx + 2*np.pi*mode[0]/d) * x + (ky + 2*np.pi*mode[1]/d) * y)
#         return np.cos(sign * phase) * r

#     def Q_im(r, theta, mode, kx):
#         x = r * np.cos(theta)
#         y = r * np.sin(theta)
#         phase = ((kx + 2*np.pi*mode[0]/d)*x + (ky + 2*np.pi*mode[1]/d) * y)
#         return np.sin(sign * phase) * r

#     # integrates over x from 0 to a, and y from 0 to a
#     q_re, _ = dblquad(Q_re, 0, 2*np.pi, lambda x: 0, lambda x: r0, args=(mode, kx))
#     q_im, _ = dblquad(Q_im, 0, 2*np.pi, lambda x: 0, lambda x: r0, args=(mode, kx))

#     return q_re + 1j * q_im

# def Q_bessel(kx, mode, sign):
#     alpha = kx + 2 * np.pi * mode[0] / d
#     beta = ky + 2 * np.pi * mode[1] / d
#     q = np.sqrt(alpha**2 + beta**2)
#     # Avoid division by zero
#     Q = np.where(np.isclose(q, 0), np.pi * a**2, 2 * np.pi * j1(sign * a * q) / q)
#     return Q

# def T_DF2(k, modes, h, kx):
#     '''equation for 0 order T from paper on holey structures'''
#     k0 = k  # np.sqrt(kx**2 + ky**2 + k**2)  # THIS PART WILL NEED RETHINKING FOR M
#     k0_prime = k0  # change if considering losses

#     a = 1.2  # circle radius

#     hg = 0.47  # 0.94 mm
#     h1 = h
#     h2 = h + hg
#     h3 = 2*h + hg

#     def S1(k, modes, kx):
#         sum1 = 0
#         for m in modes:
#             s1 = k * Q_bessel(kx, m, -1) * Q_bessel(kx, m, +1) / (d**2 * KZ(k, m, kx))
#             sum1 = sum1 + s1
#         return sum1
    
#     def S2(k, modes, kx):
#         sum2 = 0
#         for m in modes:
#             s2 = 1j * (1/np.tan(KZ(k, m, kx)*hg)) * k * Q_bessel(kx, m, -1) * Q_bessel(kx, m, +1)   \
#                 / (d**2 * KZ(k, m, kx))
#             sum2 = sum2 + s2
#         return sum2
    
#     def S3(k, modes, kx):
#         sum3 = 0
#         for m in modes:
#             s3 = 1j * (1/np.sin(KZ(k, m, kx)*hg)) * k * Q_bessel(kx, m, -1) * Q_bessel(kx, m, +1)   \
#                 / (d**2 * KZ(k, m, kx))
#             sum3 = sum3 + s3
#         return sum3

#     def D(k, m, kx):
#         '''m = modes'''
#         s1 = S1(k, m, kx)
#         s2 = S2(k, m, kx)
#         s3 = S3(k, m, kx)

#         D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi * a**2 +\
#             s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
        
#         D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi * a**2 - \
#             s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
    
#         D3 = -2 * (np.pi**2 * a**4 - s1**2) * (np.pi**2 * a**4 -       \
#             s2**2 + s3**2) / (2*np.pi**2 * a**4 * s3)
        
#         d = D1 + D2 + D3
#         return d
    
#     T0 = 0
#     # for kx in kxs:
#     t = k * 4 * Q_bessel(kx, [0, 0], +1) * Q_bessel(kx, [0, 0], -1) /  \
#         (d**2 * KZ(k, [0, 0], kx) * D(k, modes, kx))
#     T0 = T0 + t
#     return T0


# def T_dspn(m_list, k, kxs):
#     K, KX = np.meshgrid(k, kxs)
#     h = 12
# # T NEEDS TO BE A FUNCTION OF KXS - IT ALREADY HAS KX AS VARIABLE IN Q
#     Z = np.zeros((len(kxs), len(k)), dtype=complex)
#     for i, kx in enumerate(kxs):
#         for j, kv in enumerate(k):
#             Z[i, j] = T_DF2(kv, m_list, h, kx)

#     # for plotting all data (incl im) and normalisation
#     Z = np.abs(Z)
#     Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

#     plt.imshow(Z_norm, extent=[kxs.min(), kxs.max(), k.min(), k.max()],
#                             origin='lower', cmap='jet', aspect='auto')
#     plt.colorbar(label='Transmission coeff')
#     plt.title(f'Transmission for m_max={max(m_list)}')
#     plt.xlabel('Maximum Pipe Depth (mm)')
#     plt.ylabel('Frequency (Hz)')
#     # converts back to frequency (location on y, str label)
#     plt.yticks(frequency*2*np.pi/c, ['%d' % val for val in frequency])

#     plt.show()




def Q_bessel_vec(kx, mode, sign):
    # kx: 2D array, mode: tuple
    m1, m2 = mode
    ky = 0
    alpha = kx + 2 * np.pi * m1 / d
    beta = ky + 2 * np.pi * m2 / d
    q = np.sqrt(alpha**2 + beta**2)
    Q = np.where(q == 0, np.pi * a**2, 2 * np.pi * j1(sign * a * q) / q)
    return Q

def KZ_vec(k, m, kx):
    # k, kx: 2D arrays, m: tuple
    m1, m2 = m
    ky = 0
    arg = k**2 - (kx + 2*m1*np.pi/d)**2 - (ky + 2*m2*np.pi/d)**2
    return np.sqrt(arg.astype(complex))

def T_DF2_vectorized(k, kx, modes, h):
    # k, kx: 2D meshgrid arrays
    a = 1.2
    d = 8
    hg = 0.94

    # S1, S2, S3 as arrays
    S1 = np.zeros_like(k, dtype=complex)
    S2 = np.zeros_like(k, dtype=complex)
    S3 = np.zeros_like(k, dtype=complex)
    for m in modes:
        kz = KZ_vec(k, m, kx)
        q_minus = Q_bessel_vec(kx, m, -1)
        q_plus = Q_bessel_vec(kx, m, +1)
        S1 += k * q_minus * q_plus / (d**2 * kz)
        S2 += 1j * (1/np.tan(kz*hg)) * k * q_minus * q_plus / (d**2 * kz)
        S3 += 1j * (1/np.sin(kz*hg)) * k * q_minus * q_plus / (d**2 * kz)

    s1, s2, s3 = S1, S2, S3
    D1 = np.exp(-2j*k*h) * (np.pi*a**2 + s1)**2 * ((np.pi * a**2 + s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
    D2 = np.exp(+2j*k*h) * (np.pi*a**2 - s1)**2 * ((np.pi * a**2 - s2)**2 - s3**2) / (2*np.pi**2 * a**4 * s3)
    D3 = -2 * (np.pi**2 * a**4 - s1**2) * (np.pi**2 * a**4 - s2**2 + s3**2) / (2*np.pi**2 * a**4 * s3)
    D = D1 + D2 + D3

    # Numerator (for mode (0,0))
    q_plus_00 = Q_bessel_vec(kx, (0,0), +1)
    q_minus_00 = Q_bessel_vec(kx, (0,0), -1)
    kz_00 = KZ_vec(k, (0,0), kx)
    T = k * 4 * q_plus_00 * q_minus_00 / (d**2 * kz_00 * D)
    return T

def T_dspn(m_list, k, kxs):
    K, KX = np.meshgrid(k, kxs, indexing='ij')  # shape (len(k), len(kxs))
    h = 12
    Z = T_DF2_vectorized(K, KX, m_list, h)
    Z = np.abs(Z)
    Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    plt.imshow(Z_norm.T, extent=[kxs.min(), kxs.max(), k.min(), k.max()],
               origin='lower', cmap='jet', aspect='auto')
    plt.colorbar(label='Transmission coeff')
    plt.title(f'Transmission for m_max={max([max(m) for m in m_list])}')
    plt.xlabel('kx')
    plt.ylabel('k')
    plt.yticks(frequency*2*np.pi/c, ['%d' % val for val in frequency])
    plt.show()



h = 12
size = 3
m_list = [(x,x) for x in range(size)] + [(1,x) for x in range(size)] + [(x,1) for x in range(size)]
m_list = list(dict.fromkeys(m_list)) 

kxs = np.linspace(0.01, 1, N)
k0 = np.linspace(0.01, 2, N)

# thetas = [float(i) for i in range(85,95)]
# for theta in thetas:
#     kxs = np.sin((180/np.pi)*theta)*np.linspace(f1*2*np.pi/c, f2*2*np.pi/c, N)

#     T_dspn(m_list, k, kxs)

T_dspn(m_list, k0, kxs)
