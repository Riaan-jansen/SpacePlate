# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:29:52 2025\n

@author: rj429\n

Data processing experimental results\n

Start by defining experimental parameters:\n
 - time: timesteps, time samples
 - space: space steps in x, starting coordinates
 - distances: distance from plate to speaker, measurements of the plate itself
 - physical constants: speed of sound
 - frequency limits: for plotting

filename refers to data with sample, backfile refers to without. Must be real
filepaths.

###############################################################################

TO DO THINGS:
 - PLOT TRANSMISSION AMPLITUDE: plot1Damp(filename, backfile, plot=True)

 - PLOT DISPERSION RELATION: plot2Dfft(filename, backfile)

 - PLOT UNWRAPPED PHASE: plot1Dphase(filename, backfile, freq=10000, angle=False)

 - COMPRESSION FACTOR (geometric argument): compression_factor(filename, backfile)
    - option to plot fitted theoretical phases
    - plot apparent distance for plate/no plate case over all frequencies
    - plot compression factor over all frequencies

 - COMPRESSION FACTOR (Q-factor argument): compression_resonant(filename, backfile)

 - GRAPH DATA: graph_all(filename, backfile)
     for both plate/no plate at a specific frequency plots: 
    - absolute value of signal over space
    - wrapped phase
    - unwrapped phase
    - and |t|^2 over all frequencies at normal incidence

###############################################################################

CONTENTS
Functions:
 - Utilities:
    - windowing(filename): takes data object (3d array). returns 2d array after
      mean subtraction and tukey windowing

    - cleanup(filename, backfile): only works with 1D scan. load both datasets
    fft both and element-wise divide.
    Input: filepath to data, background.
    Returns 2D fft'd array.

    - search_closest(arr, multiplier=0.15):
        Search for the closest values to multiplier * max(arr) from the middle
          of the array, one on the left and one on the right.

- Plotting:
    - plot1Dxt(filename):
        given filepath, plots space vs time and signal as colourmap

    - plot1Dfft(filename):
        plots colourmap of data, time transformed to frequency against space.

    - plot1Damp(filename, backfile, plot=False):
        to plot transmission against frequency for one x. Exported to comparision
        graphs to comapre against theory and comsol. Plot = False is default,
        means it only returns two arrays to plot, doesn't plot itself.

    - plot2Dfft(filename, backfile):
        plots dispersion relation

    - plot1Dphase(filename, backfile, freq=18500, angle=False):
        argument against space @ one frequency

    - compression_factor(filename, backfile):
        calculates the compression factor using geometric argument.
        - performs a best fit of the theoretical wavefront (phase_theory()) to the
        unwrapped phase of both the space plate and free space data\n
        - does so for every frequency in the range f1, f2\n
        - returns a best fit apparent distance from the source to the mic and plots 
        that distance\n
        - takes the difference between the apparent distance for plate/without and uses
        that to calculate and plot compression factor over frequency.\n
        OPTIONAL:\n
        - can plot the fitted phases for each frequency by uncommenting code block

    - compression_resonant(filename, backfile): 
        fits a gaussian to the resonant peaks to find Q factor. Q gives
        a best theoretical estimate of C.

    - graph_all(filename, backfile, freq=10000):
        given filepath to sample and no sample data - optional: frequency 
        (default 10kHz) - returns 4 subplots:\n
        - Transmission over frequency range (range hardcoded)\n
        - Signal over space\n
        - Wrapped phase response over space\n
        - Unwrapped phase

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
x1 = 478  # for 23cm scans use 510 - later scans start further in use 478 
dx = 6.7/3
nx = int((x1 - x0) / dx) + 1

t = np.arange(0,nt)*dt
x = np.arange(x0,x1+dx,dx)

# experimental geometry parameters
d_sp = 19.6 + 1.5*2  # length of plates + gap
speaker2plate = 170
D = 230 #speaker2plate + d_sp + 20  # total distance from source to mic
c0 = 344820

# frequency limits! not defined in the functions only here
f1 = 5000; f2 = 30000

###############################################################################
################################ utility functions ############################

def windowing(data):
    '''takes data object (3d array). returns 2d array after mean subtraction
    and tukey windowing''' 
    temp = data[:,0,:]

    # plt.plot(t, temp[100,:])
    # plt.title('raw data time signal')
    # plt.xlabel('time (s)')
    # plt.ylabel('signal (V)')
    # plt.show()

    t_max = int(0.0055/dt)  # 0.006/dt is best for 21_x3

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

    # plt.plot(t, temp[100,:])
    # plt.title('windowed time signal')
    # plt.xlabel('time (s)')
    # plt.ylabel('signal (V)')
    # plt.show()
    return temp


def cleanup(filename, backfile, dspn=False):
    '''only works with 1D scan. load both datasets. fft both and element-wise divide.\n
    Input: filepath to data, background.\n
    Returns 2D fft'd array.'''
    data = np.load(filename)
    backdata = np.load(backfile)
    temp = windowing(data)
    noise = windowing(backdata)
    temp2 = data[:,0,:]
    
    # plt.plot(t, temp2[37,:], ls='--', label='w/o window and mean subtract')
    # plt.plot(t, temp[37,:])
    # plt.legend()
    # plt.xlabel('time (s)')
    # plt.ylabel('signal (V)')
    # plt.grid()
    # plt.show()

    # fft 2d
    if dspn:
        fft_temp = np.fft.fftshift(np.fft.fft2(temp))
        fft_noise = np.fft.fftshift(np.fft.fft2(noise))
    else:
        fft_temp = np.fft.fftshift(np.fft.fft(temp, axis=1), axes=1)
        fft_noise = np.fft.fftshift(np.fft.fft(noise, axis=1), axes=1)
    # element wise division
    clean_data = np.divide(fft_temp, fft_noise)

    return clean_data


def search_closest(arr, multiplier=0.15):
        """
        Search for the closest values to multiplier * max(arr) from the middle of the array,
        one on the left and one on the right.\n
        - arr: array to search.\n
        - threshold: multiplier to calculate the target value (default 0.25).\n
        returns tuple with two indexes, closest index on left and right
        """
        x = np.arange(x0,x1+dx,dx)
        x_0 = get_x(backfile)
        x_idx = np.searchsorted(x, x_0)

        max_value = max(arr)
        target_value = multiplier * max_value
        middle_index = x_idx
        
        left_index = -1
        right_index = -1
        closest_left_diff = float('inf')
        closest_right_diff = float('inf')
        
        # Search to the left of the middle
        for i in range(middle_index, -1, -1):
            diff = abs(arr[i] - target_value)
            if diff < closest_left_diff:
                closest_left_diff = diff
                left_index = i

        # Search to the right of the middle
        for i in range(middle_index, len(arr)):
            diff = abs(arr[i] - target_value)
            if diff < closest_right_diff:
                closest_right_diff = diff
                right_index = i

        return left_index, right_index


def get_x(filename, freq=10000):
    '''finds the x position where the speaker is using the unwrapped free
    space phase.\n input = background filepath.\n returns x (mm).'''
    data = np.load(filename)
    temp = data[:,0,:]
    fft_temp = np.fft.fftshift(np.fft.fft(temp, axis=1), axes=1)
    arg_data = np.angle(fft_temp)

    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))  

    idx = [i for i, x in enumerate(freq_time) if x == freq]

    slice_data = np.unwrap(arg_data[:,idx], axis=0)

    phase = (slice_data-np.max(slice_data))

    x_idx = [i for i, x in enumerate(phase) if x == 0]
    
    return x[x_idx[-1]]


###############################################################################
############################## plotting functions #############################
###############################################################################

def plot1Dxt(filename):
    '''given filepath, plots space vs time and signal as colourmap'''
    data = np.load(filename)  # data is 3d array
    temp = windowing(data)

    plt.pcolormesh(x, t, temp.T, shading='nearest')
    plt.xlabel('x (m)')
    plt.ylabel('time')
    plt.title('2D Space-time')
    plt.colorbar(label='Amplitude')
    plt.show()
    

def plot1Dfft(filename):
    '''plots colourmap of data, time transformed to frequency against space.'''
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
    
    #fft_temp = np.mean(fft_temp, axis=0)
    # fft_temp = np.clip(fft_temp, 0, 1)
    fft_temp = fft_temp[:,index1:index2]

    x_0 = get_x(backfile)
    x_idx = np.searchsorted(x, x_0)

    if plot:
        plt.scatter(freq_time, fft_temp[x_idx,:], s=6)
        plt.ylabel('Transmission Amplitude')
        plt.xlabel('frequency (Hz)')
        plt.title('Amplitude - frequency @ normal incidence')
        plt.grid()
        plt.show()

    return freq_time, fft_temp[x_idx,:]

    
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
    '''plots dispersion relation'''
    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))     
    freq_x = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * pi  # if dx is in mm this is 1/mm
    
    # fft2 of data
    fft_data = cleanup(filename, backfile, dspn=True)
    fft_data = np.abs(fft_data)**2
    fft_data = fft_data.T
    
    # frequency limits
    k1 = 0; k2 = 6
    
    clipped_data = np.clip(fft_data, 0, 1)
    
    plt.pcolormesh(freq_x, freq_time, clipped_data, cmap='plasma', shading='nearest')
    plt.ylim(bottom=f1, top=f2)
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

    x = np.arange(x0,x1+dx,dx)
    
    # plot slice along one frequency
    # [idx] of freq_time and fft_data
    slice_data = np.unwrap(arg_data[:,idx], axis=0)
    slice_noise = np.unwrap(arg_noise[:,idx], axis=0)
    # slice_data = arg_data[:,idx]
    # slice_noise = arg_noise[:,idx]

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
        plt.title(f'Phase - space @ {freq} Hz. Unwrapped')
        plt.xlabel('x (mm)')
        plt.ylabel('Phase (rad)')

    # show
    plt.legend()
    plt.grid()
    plt.show()


def compression_factor(filename, backfile):
    '''calculates the compression factor using geometric argument.
    - performs a best fit of the theoretical wavefront (phase_theory()) to the
    unwrapped phase of both the space plate and free space data\n
    - does so for every frequency in the range f1, f2\n
    - returns a best fit apparent distance from the source to the mic and plots 
    that distance\n
    - takes the difference between the apparent distance for plate/without and uses
    that to calculate and plot compression factor over frequency.\n
    OPTIONAL:\n
    - can plot the fitted phases for each frequency by uncommenting code block
    '''

    f1 = 5000; f2 = 30000
    x = np.arange(x0,x1+dx,dx)
    x_0 = get_x(backfile)
    x_idx = np.searchsorted(x, x_0)
    x = x - x_0

    # loading data and time windowing and mean subtracting
    data = np.load(filename)
    temp = windowing(data)

    noise = np.load(backfile)
    temp_noise = windowing(noise)

    # fourier transform along time axis
    fft_noise = np.fft.fftshift(np.fft.rfft(temp_noise, axis=1), axes=1)
    fft_temp = np.fft.fftshift(np.fft.rfft(temp, axis=1), axes=1)

    # taking the argument of the data
    arr_noise = np.angle(fft_noise)
    arr_data = np.angle(fft_temp)

    # frequency slicing
    freq_time = np.fft.fftshift(np.fft.rfftfreq(len(t), d=dt))
    threshold_freq = (f1, f2)
    index1 = np.searchsorted(freq_time, threshold_freq[0])#, side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1])#, side='right')

    # amplitude
    fft_clean = np.divide(fft_temp, fft_noise)

    t_clean = np.abs(fft_clean)**2
    # t_clean = t_clean/np.max(t_clean)

    # space slicing

    # threshold_x = (-220, 220)
    # x_idx1 = np.searchsorted(x, threshold_x[0])
    # x_idx2 = np.searchsorted(x, threshold_x[1])

    # x = x[x_idx1:x_idx2]

    # arr_data = arr_data[x_idx1:x_idx2,index1:index2]
    # arr_noise = arr_noise[x_idx1:x_idx2,index1:index2]
    
    freq_time = freq_time[index1:index2]

    arr_data = arr_data[:,index1:index2]
    arr_noise = arr_noise[:,index1:index2]

    def phase_theory(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 1):
            d = params[i]
            y = y + 2*pi*f/c0 * np.sqrt(d**2 + x**2)
        return (-y - np.max(-y, axis=0))

    # initialising arrays -- length(frequencies)
    z_free = np.zeros_like(freq_time)
    z_sp = np.zeros_like(freq_time)
    free_cov = np.zeros_like(freq_time)
    sp_cov = np.zeros_like(freq_time)

    # looping over frequencies
    for i, f in enumerate(freq_time):

        # slicing in space according to transmission strength
        # V = np.abs(fft_temp[:,i+index1])

        # i1, i2 = search_closest(V)

        # Vf = np.abs(fft_noise[:,i+index1])

        # i1_f, i2_f = search_closest(Vf)

        # i1_f, i2_f = i1, i2

        # plt.plot(x, V)
        # plt.plot(x[i1:i2], V[i1:i2])
        # plt.plot(x, Vf)
        # plt.plot(x[i1_f:i2_f], Vf[i1_f:i2_f])
        # plt.show()
        # sys.exit()
        # print(f)

        # phase_sp = np.unwrap(arr_data[i1:i2,i], axis=0)
        # phase_free = np.unwrap(arr_noise[i1_f:i2_f,i], axis=0)

        phase_sp = np.unwrap(arr_data[:,i], axis=0)
        phase_free = np.unwrap(arr_noise[:,i], axis=0)
        skip = False
        try:
            phase_free = (phase_free - np.max(phase_free, axis=0))
            phase_sp = (phase_sp - np.max(phase_sp, axis=0))
        except ValueError:
            skip = True

        if skip is False:
            guess = D
            # opt_free, cov_free = curve_fit(phase_theory, x[i1_f:i2_f], phase_free, p0=[guess])
            # opt_sp, cov_sp = curve_fit(phase_theory, x[i1:i2], phase_sp, p0=[guess*2])  # this guess will likely influence result
            opt_free, cov_free = curve_fit(phase_theory, x, phase_free, p0=[guess])
            opt_sp, cov_sp = curve_fit(phase_theory, x, phase_sp, p0=[guess*2]) 

            z_free[i] = opt_free
            z_sp[i] = opt_sp  # sometimes this is -ve when d is squared so shouldn't make any difference? but it might be a sign something is going wrong

            free_cov[i] = cov_free
            sp_cov[i] = cov_sp
        else:
            z_free[i] = 0
            z_sp[i] = 0

            free_cov[i] = 0
            sp_cov[i] = 0
            pass
        # opt_free, cov_free = curve_fit(phase_theory, x, phase_free, p0=[200])
        # opt_sp, cov_sp = curve_fit(phase_theory, x, phase_sp, p0=[600])

        # z_free[i] = opt_free

        # z_sp[i] = opt_sp

        # free_cov[i] = cov_free
        # sp_cov[i] = cov_sp

###############################################################################
#######               Plotting theoretical fits vs data
###############################################################################
        # fig, axs = plt.subplots(2)
        # fig.suptitle(f'Normalised Theoretical Phase vs Data @ {f:.0f} Hz')
        # axs[0].plot(x, phase_theory(x, z_sp[i]), label=f'theoretical fit d={z_sp[i]:.2f}')
        # axs[0].plot(x, phase_sp, label='spaceplate phase')
        # axs[1].plot(x, phase_theory(x, z_free[i]), label=f'theoretical fit d={z_free[i]:.2f}')
        # axs[1].plot(x, phase_free, label='free space phase')
        # axs[0].grid()
        # axs[0].legend()
        # axs[1].grid()
        # axs[1].legend()
        # plt.xlabel('x (mm)')
        # plt.show()
        # sys.exit()
###############################################################################

    tot_cov = free_cov + sp_cov
    tot_err = np.sqrt((tot_cov))

###############################################################################
####              FOR ELIMINATING VALUES OVER CERTAIN DEVIATION
###############################################################################
    # 
    # for i, ele in enumerate(tot_err):
    #     if ele >= 2.1:
    #         tot_err[i] = np.nan
    #         z_free[i] = np.nan
    #         z_sp[i] = np.nan
    #         freq_time[i] = np.nan

    L = z_sp - z_free
    C = (L + d_sp) / d_sp

###############################################################################
####              plot apparent distance to source
    plt.plot(freq_time/1000, -z_free, label='free space')
    plt.plot(freq_time/1000, -z_sp, label='spaceplate')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Apparent Distance to Source (mm)')
    plt.title('Spaceplate Compression Factor')
    plt.grid()
    plt.legend()
    plt.show()

####              plot compression factor for each frequency
    plt.scatter(freq_time/1000, C, label='spaceplate', s=7, c=tot_err)
    plt.colorbar(label='Deviation')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Compression Factor')
    plt.title('Spaceplate Compression Factor')
    #plt.xlim((8,12))
    plt.grid()
    plt.legend()
    plt.show()
###############################################################################
    return C


def compression_resonant(filename, backfile):
    ''' 
    fits a gaussian to the resonant peaks to find Q factor. Q gives
    a best theoretical estimate of C.'''
    fft_data = cleanup(filename, backfile)
    fft_data = np.abs(fft_data)**2

    # time -> frequency values for y axis
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
    fft_data = np.clip(fft_data, 0, 1)
    # clipped_temp = fft_temp

    x_0 = get_x(backfile)
    x_idx = np.searchsorted(x, x_0)

    def gaussian(x, *params):
        t = np.zeros_like(x)
        for i in range(0, len(params), 3):
            centre = params[i]
            amp = params[i+1]
            width = params[i+2]
            t = t + amp * np.exp( -((x - centre) / (np.sqrt(2)* width))**2)
        return t

    f1 = 8000; f2 = 30000
    # this slicing of frequency is necessary for best results
    index1 = np.searchsorted(freq_time, f1, side='left')
    index2 = np.searchsorted(freq_time, f2, side='right')
    freq_time = freq_time[index1:index2]
    
    # guess has lists length(params) for as many peaks as there should be present
    # each guess is a list of 3 (params)
    guess1 = [10000, 1, 10]
    guess2 = [18000, 1, 10]
    guess3 = [26000, 0.5, 10]
    #guess4 = [35000, 0.5, 10]
    guess = guess1 + guess2 + guess3 # + guess4

    # fft_data[x_idx,:] = fft_data[x_idx,:]/np.max(fft_data[x_idx,:])

    # performs the curve fit gaussian to the data - for bst results data needs cleanup and windowing
    optimised, covariance = curve_fit(gaussian, freq_time, fft_data[x_idx,index1:index2], p0=guess)

    fit = gaussian(freq_time, *optimised)

    err = np.sqrt(np.diagonal(covariance))

    colours = ['green', 'limegreen', 'darkgreen', 'lime']
    BWs = []
    centres = []

    for i in range(0, len(optimised), 3):
        sigma = optimised[i+2]
        centre = optimised[i]
        FWHM = 2*np.sqrt(2*np.log(2)) * sigma

        # EQUATION FROM SPACE SQUEEZING OPTICS PAPER
        Q = centre/FWHM
        c = Q/(2*((i/3)+1))

        centres.append(centre)
        BWs.append(FWHM)

        plt.axvline(centre, color=colours[int(i/3)], ls=':', label=f'{centre:.0f} Hz, C = {c:.2f}')

    plt.plot(freq_time, fft_data[x_idx,index1:index2])
    plt.plot(freq_time, fit,  color='red', ls='--')
    plt.ylabel('Transmission Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.title('Compression Factor via Resonances @ normal incidence')
    plt.grid()
    plt.legend()
    plt.show()

    C = []
    C_errs = []
    for i in range(len(centres)):
        Q = centres[i]/BWs[i]
        c = Q/(2*(i+1))
        C.append(c)

        # from cov matrix
        centre_err = err[i*3]
        FWHM_err = 2*np.sqrt(2*np.log(2)) * err[i*3+2]

        # errors added in quadrature
        Q_err = Q * np.sqrt((centre_err / centres[i])**2 + (FWHM_err / BWs[i])**2)
        C_err = Q_err / (2 * (i + 1))
        C_errs.append(C_err)

        print(f'C = {c:.2f} +/- {C_err:.2f} @ f = {centres[i]:.0f} Hz')

    return C, C_errs


def graph_all(filename, backfile, freq=10000):
    '''given filepath to sample and no sample data - optional: frequency 
    (default 10kHz) - returns 4 subplots:\n
    - Transmission over frequency range (range hardcoded)\n
    - Signal over space\n
    - Wrapped phase response over space\n
    - Unwrapped phase'''
    x = np.arange(x0,x1+dx,dx)
    x_0 = get_x(backfile, freq)
    x_idx = np.searchsorted(x, x_0)
    x = x - x_0

    # loading data and time windowing and mean subtracting
    data = np.load(filename)
    temp = windowing(data)

    noise = np.load(backfile)
    temp_noise = windowing(noise)

    # fourier transform along time axis
    fft_noise = np.fft.fftshift(np.fft.fft(temp_noise, axis=1), axes=1)
    fft_temp = np.fft.fftshift(np.fft.fft(temp, axis=1), axes=1)

    # amplitude
    t_noise = np.abs(fft_noise)**2
    t_temp = np.abs(fft_temp)**2
    t_clean = np.divide(t_temp, t_noise)

    # taking the argument of the data
    arr_noise = np.angle(fft_noise)
    arr_data = np.angle(fft_temp)

    # frequency slicing
    freq_time = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))

    threshold_freq = (5000, 30000)
    
    index1 = np.searchsorted(freq_time, threshold_freq[0])#, side='left')
    index2 = np.searchsorted(freq_time, threshold_freq[1])#, side='right')
    freq_time = freq_time[index1:index2]
    # frequency to plot for
    idx = np.searchsorted(freq_time, freq)

    # t_clean = t_clean/np.max(t_clean)

    # plt.plot(x, t_clean[:,index1])
    # plt.show()

    # space slicing 
    # threshold_x = (-200, 200)
    # x_idx1 = np.searchsorted(x, threshold_x[0])
    # x_idx2 = np.searchsorted(x, threshold_x[1])
    # x = x[x_idx1:x_idx2]

    # arr_data = arr_data[x_idx1:x_idx2,index1:index2]
    # arr_noise = arr_noise[x_idx1:x_idx2,index1:index2]

    arr_data = arr_data[:,index1:index2]
    arr_noise = arr_noise[:,index1:index2]

    # unwarpped phase
    phase_sp = np.unwrap(arr_data[:,idx], axis=0)
    phase_free = np.unwrap(arr_noise[:,idx], axis=0)

    phase_free = (phase_free - np.max(phase_free, axis=0))
    phase_sp = (phase_sp - np.max(phase_sp, axis=0))

    # wrapped phase
    wphase_free = (arr_noise[:,idx] - np.max(arr_noise[:,idx], axis=0))
    wphase_sp = (arr_data[:,idx] - np.max(arr_data[:,idx], axis=0))

    # plt.plot(x, np.abs(fft_temp[:,idx+index1]), label='spaceplate')
    # plt.plot(x, np.abs(fft_noise[:,idx+index1]), label='free space')
    # plt.show()
    # sys.exit()

    fig, axs = plt.subplots(4)
    fig.suptitle(f'Spaceplate Data @ {freq:.0f} Hz')

    axs[0].plot(freq_time, t_clean[x_idx,index1:index2], label ='$|t|^2$')
    axs[0].set_ylabel(r'$|t|^2$')
    axs[0].set_xlabel('Frequency (Hz)')

    axs[1].plot(x, np.abs(fft_temp[:,idx+index1]), label='spaceplate')
    axs[1].plot(x, np.abs(fft_noise[:,idx+index1]), label='free space')
    axs[1].set_ylabel('$|t|$')

    axs[2].plot(x, wphase_sp, label='spaceplate phase')
    axs[2].plot(x, wphase_free, label='free space phase')
    axs[2].set_ylabel('Wrapped Phase')

    axs[3].plot(x, phase_sp, label='spaceplate phase')
    axs[3].plot(x, phase_free, label='free space phase')
    axs[3].set_ylabel('Unwrapped Phase')

    axs[0].grid()
    axs[0].legend()
    axs[1].grid()
    axs[1].legend()
    axs[2].grid()
    axs[2].legend()
    axs[3].grid()
    axs[3].legend()
    plt.xlabel('x (mm)')

    plt.show()
    return

###############################################################################
################################ calling ######################################
if __name__ == '__main__':
    #           ******************** FILENAMES *************************
    # filename = 'scan_sp55_2.npy'
    # backfile = 'scan_ns55_2.npy'

    filename = 'scan_data/scan_sp21_x3.npy'
    backfile = 'scan_data/scan_ns21_x3.npy'

    # filename = 'scan_sp21_2.npy'
    # backfile = 'scan_ns21.npy'
    #     *****************************************************************
    #      ******************** TRANSMISSION AMPLITUDE *************************

    # plot1Damp(filename, backfile, plot=True)

    # plot1Dfft(filename)

    #     *****************************************************************
    #      ******************** DISPERSION RELATION *************************

    # plot2Dfft(filename, backfile)

    #     *****************************************************************
    #      ******************** PHASE PROFILE ****************************

    # plot1Dphase(filename, backfile, freq=8843.75)

    #     *****************************************************************
    #      ******************** COMPRESSION FACTOR ************************
    
    compression_resonant(filename, backfile)  # based on resonance argument
    # compression_factor(filename, backfile)    # based on geometric argument

    #     *****************************************************************
    #      *********************** BIT OF EVERYTHING ************************
    
    # graph_all(filename, backfile, freq=10500)

    #     *****************************************************************
    #    *********************** CHECKING THE DATA ************************

    # cleanup(filename, backfile)

    # plot1Dxt(backfile)

    # data = np.load(filename)
    # windowing(data)
