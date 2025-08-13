# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:29:52 2025

@author: rj429

Data processing experimental results
"""

import numpy as np
from numpy import pi
from scipy import signal
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

import sys


###############################################################################
############################## experimental params ############################

# defining t and x arrays from the scan settings
dt = 3.2e-6
nt = 10000

x0 = 0
x1 = 510
dx = 6.7
nx = int((x1 - x0) / dx) + 1

t = np.arange(0,nt)*dt
x = np.arange(x0,x1+dx,dx)

# experimental geometry parameters
d_sp = 19.6 + 1.5*2
D = 230 + d_sp + 23  # to plate, sp, plate to mic
c0 = 343000

# frequency limits! not defined in the functions only here
f1 = 5000; f2 = 30000

###############################################################################
######################### functions for processing ############################

def windowing(filename):
    data = np.load(filename)

    temp = data[:,0,:]

    t_max = int(0.01/dt)

    tukey = signal.tukey(t_max, alpha=0.5)
    tukey_window = np.pad(tukey, (0, max(0, nt - t_max)), mode='constant', 
                          constant_values=0)
    
    # tukey(length=point up until you want to have signal) + zeroes after
    # tukey * both temp and noise in loops v

    # mean subtraction of the mean of the signal along time for each x step
    temp_mean = np.mean(temp, axis=1)
    for i, mean in enumerate(temp_mean):
        temp[i,:] = temp[i,:] - mean

    temp = np.multiply(temp, tukey_window)
    return temp

def cleanup(filename, backfile):
    '''only works with 1D scan. load both datasets, mean subtraction and Tukey
    windowing, fft both and element-wise divide.\n
    Input: filepath to data, background.\n
    Returns 2D fft'd array.'''
    data = np.load(filename)
    back = np.load(backfile)
    
    temp = data[:,0,:]
    noise = back[:,0,:]

    t_max = int(0.01/dt)

    tukey = signal.tukey(t_max, alpha=0.5)
    tukey_window = np.pad(tukey, (0, max(0, nt - t_max)), mode='constant', 
                          constant_values=0)
    
    # tukey(length=point up until you want to have signal) + zeroes after
    # tukey * both temp and noise in loops v

    # mean subtraction of the mean of the signal along time for each x step
    temp_mean = np.mean(temp, axis=1)
    for i, mean in enumerate(temp_mean):
        temp[i,:] = temp[i,:] - mean

    back_mean = np.mean(noise, axis=1)
    for i, mean in enumerate(back_mean):
        noise[i,:] = noise[i,:] - mean

    noise = np.multiply(noise, tukey_window)
    temp = np.multiply(temp, tukey_window)

    # plt.plot(t, noise[0,:])
    # plt.plot(t, noise2[0,:])
    # plt.show()

    # fft 2d
    fft_temp = np.fft.fftshift(np.fft.fft2(temp))
    fft_noise = np.fft.fftshift(np.fft.fft2(noise))
    # element wise division
    clean_data = np.divide(fft_temp, fft_noise)

    return clean_data

# backfile = r'scan_nosample18.npy'
# backfile = 'scan_sample18.npy'
# back = np.load(backfile)
# noise = back[:,0,:]
# fft_noise = np.fft.fftshift(np.fft.fft2(noise))
# fft_temp = np.abs(fft_noise)**2

# # time -> frequency values for y axis
# freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
# threshold_freq = (1000, 40000)
# index1 = np.searchsorted(freq_time, threshold_freq[0], side='left')
# index2 = np.searchsorted(freq_time, threshold_freq[1], side='right')

# freq_time = freq_time[index1:index2]

# # clipped_temp = np.clip(fft_temp, 0, 1)
# clipped_temp = fft_temp

# x0 = 288.1  # 18cm scan
# dx = 6.7
# x_idx = int(x0/dx)

# plt.plot(freq_time, clipped_temp[x_idx,index1:index2])
# plt.ylabel('Transmission Amplitude')
# plt.xlabel('frequency (Hz)')
# plt.title('Amplitude - frequency @ normal incidence')
# plt.grid()
# plt.show()

def plot1Dxt(filename):
    '''given filepath, plots space vs time and signal as colourmap'''
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

    clipped_temp = np.clip(fft_temp, 0, 5000)
    # clipped_temp = fft_temp
    
    plt.pcolormesh(x, freq_time, clipped_temp, shading='nearest', cmap='plasma')
    plt.ylim(bottom=f1, top=f2)
    plt.xlabel('x (mm)')
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
    threshold_freq = (f1, f2)
    index1 = np.searchsorted(freq_time, threshold_freq[0], side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1], side='right')
    
    freq_time = freq_time[index1:index2]

    clipped_temp = np.clip(fft_temp, 0, 1)
    # clipped_temp = fft_temp

    x0 = 288.1  # 18cm scan
    dx = 6.7
    x_idx = int(x0/dx)
    if plot:
        plt.plot(freq_time, clipped_temp[x_idx,index1:index2])
        plt.ylabel('Transmission Amplitude')
        plt.xlabel('frequency (Hz)')
        plt.title('Amplitude - frequency @ normal incidence')
        plt.grid()
        plt.show()

    return freq_time, clipped_temp[x_idx,index1:index2]

    
def plot2Dfft_old(filename):
    '''for immediate plotting without background sample'''
    # load data and transpose: columns -> rows
    data = np.load(filename)  # data is 3d array
    temp = data[:,0,:].T

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     
    freq_x = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * pi  # if dx is in mm this is 1/mm
    
    # fft2 of data
    fft_temp = np.fft.fftshift(np.fft.fft2(temp))
    fft_temp = np.abs(fft_temp)**2
    
    # frequency limits
    k1 = 0; k2 = +6  # NEEDS MORE (variable)
    
    clipped_temp = np.clip(fft_temp, 0, 1000)

    plt.pcolormesh(freq_x, freq_time, clipped_temp, shading='nearest')
    plt.ylim(bottom=f1, top=f2)
    plt.xlim(left=k1)
    plt.xlabel('$k_x$ (1/mm)')
    plt.ylabel('frequency (Hz)')
    plt.title('2D freq - k-space')
    plt.colorbar(label='Amplitude')
    plt.show()


def plot2Dfft(filename, backfile):
    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     
    freq_x = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * pi  # if dx is in mm this is 1/mm
    
    # fft2 of data
    fft_data = cleanup(filename, backfile)
    fft_data = np.abs(fft_data)**2
    fft_data = fft_data.T
    
    # frequency limits
    y1 = 0; y2 = 40000
    k1 = 0; k2 = 6
    
    clipped_data = np.clip(fft_data, 0, 1)
    
    plt.pcolormesh(freq_x, freq_time, clipped_data, cmap='plasma', shading='nearest')
    plt.ylim(bottom=y1, top=y2)
    plt.xlim(left=k1)
    plt.xlabel('$k_x$ (1/mm)')
    plt.ylabel('frequency (Hz)')
    plt.title('Spaceplate Dispersion Relation - Experiment')
    plt.colorbar(label='Transmission Amplitude')
    plt.show()

###############################################################################
##################### phase and compression factor ############################

def plot2Dphase(filename, backfile):
    '''frequency against space, plotting the argument of the data'''
    fft_data = cleanup(filename, backfile)
    fft_data = np.angle(fft_data)
    fft_data = fft_data.T

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     
    
    plt.pcolormesh(x, freq_time, fft_data, cmap='plasma', shading='nearest')
    plt.ylim(bottom=f1, top=f2)
    plt.xlabel('x (m)')
    plt.ylabel('frequency (Hz)')
    plt.title('Phase - frequency - x')
    plt.colorbar(label='Phase')
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

    # from space squeezing optics paper
    # phase_diff = slice_data - slice_noise

    phase_diff = phase_sp - phase_free

    d_eff = phase_diff / (2*pi * freq * np.cos(theta) / c0)
    
    C = np.mean(d_eff) / d_sp

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
    
    print(f'compression factor @ {freq} Hz = {C}')
    return C


def C_factor(filename, backfile):
    '''plotting compression factor for all frequencies'''

    # loading data
    data = np.load(filename)
    temp = data[:,0,:]
    fft_temp = np.fft.fftshift(np.fft.rfft(temp, axis=1), axes=1)
    arg_data = np.angle(fft_temp)
    
    noise = np.load(backfile)
    temp_noise = noise[:,0,:]
    fft_noise = np.fft.fftshift(np.fft.rfft(temp_noise, axis=1), axes=1)
    arg_noise = np.angle(fft_noise)

    # arrays
    x = np.arange(x0,x1+dx,dx)
    freq_time = np.fft.fftshift(np.fft.rfftfreq(len(t), d=dt))

    # slicing sensible frequency range
    threshold_freq = (5000, 40000)
    index1 = np.searchsorted(freq_time, threshold_freq[0])#, side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1])#, side='right')
    
    freq_time = freq_time[index1:index2]
    arg_data = arg_data[:,index1:index2]
    arg_noise = arg_noise[:,index1:index2]

    # measurements of experimental setup (mm)
    d_sp = 19.6 + 1.5*2
    D = 180 + d_sp + 25  # to plate, sp, plate to mic
    c0 = 343000

    C_list = []
    
    for i, f in enumerate(freq_time):
    # calculate C for every frequency
        slice_data = np.unwrap(arg_data[:,i], axis=0)
        slice_noise = np.unwrap(arg_noise[:,i], axis=0)

        phase_free = (slice_noise-np.max(slice_noise, axis=0))
        phase_sp = (slice_data-np.max(slice_data, axis=0))

        x_idx = [i for i, x in enumerate(phase_free) if x == 0]

        # try:
        #     x = x - x[x_idx]  # shift to center on theta=0
        # except ValueError:
        #     pass  
        x = x - x[x_idx]

        # calculating compression factor from space squeezing optics paper
        theta = (np.arctan(x/D))

        phase_diff = phase_sp - phase_free

        d_eff = phase_diff / (2*pi * f * np.cos(theta) / c0)

        C = np.mean(d_eff) / d_sp
        C_list.append(C)
        
    plt.scatter(freq_time/1000, C_list, s=2.5)
    plt.title('Spaceplate Compression Factors')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Compression factor')
    plt.grid()
    plt.show()


def compression_factor(filename, backfile):
    '''calculates the compression factor using geometric argument'''
    x = np.arange(x0,x1+dx,dx)
    x = x - 288.1

    # range of distances to try (speaker to mic)
    d_range = np.arange(100,600,1)

    data = np.load(filename)
    temp = data[:,0,:]

    noise = np.load(backfile)
    temp_noise = noise[:,0,:]

    # time windowing
    t_max = int(0.01/dt)
    tukey = signal.tukey(t_max, alpha=0.5)
    tukey_window = np.pad(tukey, (0, max(0, nt - t_max)), mode='constant', 
                          constant_values=0)

    temp = np.multiply(temp, tukey_window)
    temp_noise = np.multiply(temp_noise, tukey_window)

    # fourier transform along time axis
    fft_noise = np.fft.fftshift(np.fft.rfft(temp_noise, axis=1), axes=1)
    arr_noise = np.angle(fft_noise)
    fft_temp = np.fft.fftshift(np.fft.rfft(temp, axis=1), axes=1)
    arr_data = np.angle(fft_temp)

    # slicing
    freq_time = np.fft.fftshift(np.fft.rfftfreq(len(t), d=dt))
    threshold_freq = (5000, 15000)
    index1 = np.searchsorted(freq_time, threshold_freq[0])#, side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1])#, side='right')

    threshold_x = (-100, 100)
    x_idx1 = np.searchsorted(x, threshold_x[0])
    x_idx2 = np.searchsorted(x, threshold_x[1])

    x = x[x_idx1:x_idx2]
    freq_time = freq_time[index1:index2]
    arr_data = arr_data[x_idx1:x_idx2,index1:index2]
    arr_noise = arr_noise[x_idx1:x_idx2,index1:index2]

    # looping over frequencies
    z_free = np.zeros_like(freq_time)
    z_sp = np.zeros_like(freq_time)
    for i, f in enumerate(freq_time):
        phase_sp = np.unwrap(arr_data[:,i], axis=0)
        phase_free = np.unwrap(arr_noise[:,i], axis=0)

        phase_free = (phase_free-np.max(phase_free, axis=0))
        phase_sp = (phase_sp-np.max(phase_sp, axis=0))

        diff_free = np.zeros_like(d_range)  # empties for every f
        diff_sp = np.zeros_like(d_range)
        index = 0
        for d in d_range:
            phase_theory = 2*pi*f/c0 * np.sqrt(d**2 + x**2)
            phase_theory = (-phase_theory - np.max(-phase_theory, axis=0))  # set to max at y=0 and invert to match the shape of the data
            # phase_theory = (phase_theory - np.max(phase_theory, axis=0))

            res_free = ( (phase_free - phase_theory) - np.mean(phase_free - phase_theory) )**2
            res_sp = ( (phase_sp - phase_theory) - np.mean(phase_sp - phase_theory) )**2

            #print(res_free, 'res_free') 

            # plt.plot(x, phase_theory, label='theoretical phase')
            # plt.plot(x, phase_free, label='measured free space phase')
            # plt.grid()
            # plt.legend()
            # plt.title(f'Normalised Phase theory vs data d={d}')
            # plt.xlabel('x (mm)')
            # plt.ylabel('Phase')
            # plt.show()
            # sys.exit()

            diff_free[index] = np.sqrt(np.mean(res_free))
            #print(diff_free, 'diff_free')
            diff_sp[index] = np.sqrt(np.mean(res_sp))
            index = index + 1

        idx_free = np.argmin(diff_free)
        z_free[i] = d_range[idx_free]

        idx_sp = np.argmin(diff_sp)
        z_sp[i] = d_range[idx_sp]

        # plt.plot(d_range, diff_free)
        # plt.plot(x, res_free)
        # plt.show()
        # sys.exit()

        # check_theory = 2*pi*f/c0 * np.sqrt(z_free[i]**2 + x**2)
        # check_theory = (-check_theory - np.max(-check_theory, axis=0))
        # plt.plot(x, check_theory, label='theoretical phase')
        # plt.plot(x, phase_free, label='measured free space phase')
        # plt.grid()
        # plt.legend()
        # plt.title(f'Normalised Phase theory vs data d={z_free[i]}')
        # plt.xlabel('x (mm)')
        # plt.ylabel('Phase')
        # plt.show()
        # sys.exit()

    L = z_sp - z_free
    C = (L + d_sp) / d_sp

    plt.plot(freq_time/1000, -z_free, label='free space')
    plt.plot(freq_time/1000, -z_sp, label='spaceplate')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Apparent Distance to Source (mm)')
    plt.title('Spaceplate Compression Factor')
    plt.grid()
    plt.legend()
    plt.show()

    plt.scatter(freq_time/1000, C, label='spaceplate', s=2.5)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Compression Factor')
    plt.title('Spaceplate Compression Factor')
    #plt.xlim((8,12))
    plt.grid()
    plt.legend()
    plt.show()

    return C


def compression_resonant(filename, backfile):
    '''fits a lorentzian to the resonant peaks to find Q factor. Q gives
    a best theoretical estimate of C.'''
    fft_data = cleanup(filename, backfile)
    fft_data = np.abs(fft_data)**2

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
    fft_data = np.clip(fft_data, 0, 1)
    # clipped_temp = fft_temp

    x0 = 288.1  # 18cm scan
    dx = 6.7
    x_idx = int(x0/dx)

    def gaussian(x, *params):
        t = np.zeros_like(x)
        for i in range(0, len(params), 3):
            centre = params[i]
            amp = params[i+1]
            width = params[i+2]
            t = t + amp * np.exp( -((x - centre) / (np.sqrt(2)* width))**2)
        return t

    index1 = np.searchsorted(freq_time, 8000, side='left')
    index2 = np.searchsorted(freq_time, 36000, side='right')
    
    freq_time = freq_time[index1:index2]
    
    # guess has lists length(params) for as many peaks as there should be present
    # each guess is a list of 3 (params)
    guess1 = [10000, 1, 10]
    guess2 = [18000, 1, 10]
    guess3 = [26000, 0.5, 10]
    guess4 = [35000, 0.5, 10]
    guess = guess1 + guess2 + guess3 + guess4

    fft_data[x_idx,:] = fft_data[x_idx,:]/np.max(fft_data[x_idx,:])

    optimised, covariance = curve_fit(gaussian, freq_time, fft_data[x_idx,index1:index2], p0=guess)

    fit = gaussian(freq_time, *optimised)

    colours = ['green', 'limegreen', 'darkgreen', 'lime']
    BWs = []
    centres = []
    for i in range(0, len(optimised), 3):
        sigma = optimised[i+2]
        centre = optimised[i]
        FWHM = 2*np.sqrt(2*np.log(2))*sigma
        centres.append(centre)
        BWs.append(FWHM)
        plt.axvline(centre, color=colours[int(i/3)], ls=':', label=f'{centre:.0f} Hz, FWHM = {FWHM:.2f}')

    plt.plot(freq_time, fft_data[x_idx,index1:index2])
    plt.plot(freq_time, fit,  color='red', ls='-')
    plt.ylabel('Transmission Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.title('Amplitude - frequency @ normal incidence')
    plt.grid()
    plt.legend()
    plt.show()

    C = []
    for i in range(len(centres)):
        Q = centres[i]/BWs[i]
        c = Q/(2*(i+1))
        C.append(c)

        print(f'C = {c:.2f} @ f = {centres[i]:.0f} Hz')


###############################################################################
################################ calling ######################################

# filename = r'scan_1d_sample.npy'
# backfile = r'scan_1d_nosample.npy'

# filename = r'scan_sample11.npy'
# backfile = r'scan_nosample11.npy'

filename = r'scan_sample18.npy'
backfile = r'scan_nosample18.npy'

# compression_resonant(filename, backfile)
compression_factor(filename, backfile)

#cleanup(filename, backfile)
# plot2Dfft(filename, backfile)
#plot1Damp(filename, backfile, plot=True)
# plot1Dfft(filename)
#plot1Dphase(filename, backfile, freq=10500)
# find_freq(filename, backfile)
# C_factor(filename, backfile)
