# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:29:52 2025

@author: rj429

Data processing experimental results
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


###############################################################################
############################## data processing ################################

# defining t and x arrays from the scan settings
dt = 3.2e-6
nt = 10000

x0 = 0
x1 = 510
dx = 6.7
nx = int((x1 - x0) / dx) + 1

t = np.arange(0,nt)*dt
x = np.arange(x0,x1+dx,dx)*1e-3

###############################################################################
######################### functions for processing ############################

def cleanup(filename, backfile):
    '''only works with 1D scan. load both datasets, fft both and element-wise
    divide.\n
    Input: filepath to data, background.\n
    Returns 2D fft'd array.'''
    data = np.load(filename)
    back = np.load(backfile)
    
    temp = data[:,0,:]
    noise = back[:,0,:]

    # mean subtraction
    temp = temp + np.mean(temp, axis=0)
    noise = noise + np.mean(noise, axis=0)

    # fft 2d
    fft_temp = np.fft.fftshift(np.fft.fft2(temp))
    fft_noise = np.fft.fftshift(np.fft.fft2(noise))
    # element wise division
    clean_data = np.divide(fft_temp, fft_noise)

    return clean_data
    

def plot1Dxt(filename):
    '''given filepath, plots space vs time and signal as colour values'''
    data = np.load(filename)  # data is 3d array

    temp = data[:,0,:].T

    plt.pcolormesh(x, t, temp, shading='nearest')
    plt.xlabel('x (m)')
    plt.ylabel('time')
    plt.title('2D Space-time')
    plt.colorbar(label='Amplitude')
    plt.show()
    

def plot1Dfft(filename):
    # load data and transpose: columns -> rows
    data = np.load(filename)  # data is 3d array
    temp = data[:,0,:].T
    
    # t = np.arange(ps.noSamples)*(ps.timebase-2)/(125*1e6)
    
    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
    
    # fft of the data f(t) -> F(w)
    fft_temp = np.fft.fftshift(np.fft.fft(temp, axis=0), axes=0)
    fft_temp = np.abs(fft_temp)**2
    
    # limits to frequency plotting
    y1 = 0
    y2 = 40000

    clipped_temp = np.clip(fft_temp, 0, 5000)
    # clipped_temp = fft_temp
    
    plt.pcolormesh(x, freq_time, clipped_temp, shading='nearest', cmap='plasma')
    plt.ylim(bottom=y1, top=y2)
    plt.xlabel('x (m)')
    plt.ylabel('frequency (Hz)')
    plt.title('2D freq - space')
    plt.colorbar(label='Amplitude')
    plt.show()


def plot1Damp(filename, backfile, plot=False):
    '''to plot transmission against frequency for one x. Exported to comparision
    graphs to comapre against theory and comsol.'''
    # load data and transpose: columns -> rows
    fft_temp = cleanup(filename, backfile)
    fft_temp = np.abs(fft_temp)**2

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
    threshold_freq = (10000, 40000)
    index1 = np.searchsorted(freq_time, threshold_freq[0], side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1], side='right')
    
    freq_time = freq_time[index1:index2]

    clipped_temp = np.clip(fft_temp, 0, 1)
    # clipped_temp = fft_temp

    x0 = 250
    dx = 6.75
    x_idx = int(x0/dx)
    if plot:
        plt.plot(freq_time, clipped_temp[x_idx,index1:index2])
        plt.ylabel('Transmission Amplitude')
        plt.xlabel('frequency (Hz)')
        plt.title('Amplitude - frequency @ normal incidence')
        plt.grid()
        plt.show()

    return freq_time, clipped_temp[x_idx,index1:index2]

    
def plot2Dfft(filename):
    # load data and transpose: columns -> rows
    data = np.load(filename)  # data is 3d array
    temp = data[:,23,:].T
    
    a = 0.008  # lattice constant (pitch)
    k_bz = pi / a

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     
    freq_x = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * pi  # if dx is in mm this is 1/mm
    
    # normalised reciprocal space
    k_norm = freq_x / k_bz
    
    # fft2 of data
    fft_temp = np.fft.fftshift(np.fft.fft2(temp))
    fft_temp = np.abs(fft_temp)**2
    
    # frequency limits
    y1 = 0; y2 = 40000
    # reciprocal space limited to reduced brillouin zone
    k1 = 0; k2 = +6  # NEEDS MORE (variable)
    
    clipped_temp = np.clip(fft_temp, 0, 1000)
    
    plt.pcolormesh(freq_x, freq_time, clipped_temp, shading='nearest')
    plt.ylim(bottom=y1, top=y2)
    plt.xlim(left=k1)
    plt.xlabel('$k_x$ (1/mm)')
    plt.ylabel('frequency (Hz)')
    plt.title('2D freq - k-space')
    plt.colorbar(label='Amplitude')
    plt.show()


def plot2Dfft_clean(filename, backfile):

    a = 6.7  # lattice constant (pitch)
    k_bz = pi / a

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     
    freq_x = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * pi  # if dx is in mm this is 1/mm
    
    # normalised reciprocal space
    k_norm = freq_x / k_bz
    
    # fft2 of data
    fft_data = cleanup(filename, backfile)
    fft_data = np.abs(fft_data)**2
    fft_data = fft_data.T
    
    # frequency limits
    y1 = 0; y2 = 40000
    # reciprocal space limited to reduced brillouin zone
    k1 = 0; k2 = +6  # NEEDS MORE (variable)
    
    clipped_data = np.clip(fft_data, 0, 1)
    
    plt.pcolormesh(freq_x, freq_time, clipped_data, cmap='plasma', shading='nearest')
    plt.ylim(bottom=y1, top=y2)
    plt.xlim(left=k1)
    plt.xlabel('$k_x$ (1/mm)')
    plt.ylabel('frequency (Hz)')
    plt.title('2D freq - k-space')
    plt.colorbar(label='Amplitude')
    plt.show()


def plot2Dphase(filename, backfile):
    '''frequency against space, plotting the argument of the data'''
    fft_data = cleanup(filename, backfile)
    fft_data = np.angle(fft_data)
    fft_data = fft_data.T

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     

    # frequency limits
    y1 = 0; y2 = 40000
    
    plt.pcolormesh(x, freq_time, fft_data, cmap='plasma', shading='nearest')
    plt.ylim(bottom=y1, top=y2)
    plt.xlabel('x (m)')
    plt.ylabel('frequency (Hz)')
    plt.title('Phase - frequency - x')
    plt.colorbar(label='Phase')
    plt.show()


def find_freq(filename, backfile):
    '''Input: data and background data file paths\n
    Returns: frequency at 80/20 T/R'''
    fft_data = cleanup(filename, backfile)
    fft_data = np.abs(fft_data)**2
    fft_data = fft_data
    
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
    
    threshold_freq = (1000, 40000)
    index1 = np.searchsorted(freq_time, threshold_freq[0], side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1], side='right')
    
    freq_time = freq_time[index1:index2]
    
    x0 = 0.25  # the x-position you are interested in (for scan_1d it is ~10)
    threshold = 0.2
    
    data = fft_data.T
    # find the index closest to x0
    x_idx = np.argmin(np.abs(x - x0))
    
    # normalize transmission at x0
    T_x0 = data[index1:index2, x_idx]
    T_norm = T_x0 / np.max(T_x0)
    
    # find frequency closest to t = threshold
    target_idx = np.argmin(np.abs(T_norm - threshold))
    
    # find frequency at that transmission index
    if target_idx:
        f0 = freq_time[target_idx]
    else:
        print(f"No frequency found near desired transmission.")
    
    # plot
    plt.plot(freq_time, T_norm)
    plt.axhline(threshold, color='r', linestyle='--', label=f'{threshold:.1f} T')
    plt.axvline(f0, color='g', linestyle='--', label=f'Freq = {f0:.1f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Transmission')
    plt.title(f'Transmission - frequency @ x = {x0} m')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot1Dphase(filename, backfile, freq=18500, angle=False):
    '''argument against space @ one frequency'''
    data = np.load(filename)
    temp = data[:,0,:]
    # fft_temp = np.fft.fftshift(np.fft.fft2(temp))
    fft_temp = np.fft.fftshift(np.fft.fft(temp, axis=1), axes=1)
    arg_data = np.angle(fft_temp)
    
    noise = np.load(backfile)
    temp_noise = noise[:,0,:]
    # fft_noise = np.fft.fftshift(np.fft.fft2(temp_noise))
    fft_noise = np.fft.fftshift(np.fft.fft(temp_noise, axis=1), axes=1)
    arg_noise = np.angle(fft_noise)
    
    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))  
    
    idx = [i for i, x in enumerate(freq_time) if x == freq]

    # frequency limits
    y1 = 0; y2 = 40000
    x = np.arange(x0,x1+dx,dx)
    
    d_sp = 19.6 + 1.5*2
    D = 180 + d_sp + 25  # to plate, sp, plate to mic
    c0 = 343000
    
    # plot slice along one frequency
    # [idx] of freq_time and fft_data
    slice_data = np.unwrap(arg_data[:,idx], axis=0)
    slice_noise = np.unwrap(arg_noise[:,idx], axis=0)

    # amplitude
    fft_clean = np.divide(fft_temp, fft_noise)

    t_data = np.abs(fft_temp)**2
    t_noise = np.abs(fft_noise)**2
    # t_clean = np.abs(fft_clean)**2

    # plt.plot(x_sft, t_data[:,idx], label='sample')
    # plt.plot(x_sft, t_noise[:,idx], label='no sample')
    # plt.plot(x_sft, t_clean[:,idx], label ='sample/no sample')

    # plt.ylabel('Fourier amplitude (abs(t)^2)')
    # plt.xlabel('x (shifted) (mm)')
    # plt.grid()
    # plt.legend()
    # plt.show()

    phase_free = (slice_noise-np.max(slice_noise))
    phase_sp = (slice_data-np.max(slice_data))

    x_idx = [i for i, x in enumerate(phase_free) if x == 0]
    x = x - x[x_idx]  # shift to center on theta=0

    theta = (np.arctan(x/D))
    if angle:
        # phase - angle
        plt.plot(theta*180/pi, phase_sp*180/pi, label='sample')
        plt.plot(theta*180/pi, phase_free*180/pi, label='no sample')
        plt.title(f'Phase - angle @ {freq} Hz')
        plt.xlabel('Angle (deg)')
        plt.ylabel('Phase (deg)')
    else:
        # phase - space
        plt.plot(x, phase_sp, label='sample')
        plt.plot(x, phase_free, label='no sample')
        plt.title(f'Phase - space @ {freq} Hz')
        plt.xlabel('x (mm)')
        plt.ylabel('Phase (rad)')

    # show
    plt.legend()
    plt.grid()
    plt.show()
    
    # from space squeezing optics paper
    # phase_diff = slice_data - slice_noise

    phase_diff = (slice_data-np.max(slice_data, axis=0)) - (slice_noise-np.max(slice_noise, axis=0))

    d_eff = phase_diff / (2*pi * freq * np.cos(theta) / c0)
    
    C = np.mean(d_eff) / d_sp
    print(f'compression factor = {C}')

###############################################################################
################################ calling ######################################

# filename = r'scan_1d_sample.npy'
# backfile = r'scan_1d_nosample.npy'

# filename = r'scan_sample11.npy'
# backfile = r'scan_nosample11.npy'

filename = r'scan_sample18.npy'
backfile = r'scan_nosample18.npy'

# plot2Dfft_clean(filename, backfile)
# plot1Damp(filename, backfile, plot=True)
# plot1Dfft(filename)
plot1Dphase(filename, backfile, freq=27500)
# find_freq(filename, backfile)
