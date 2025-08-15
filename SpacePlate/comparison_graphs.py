'''
For producing graphs to compare theory, numerical and experimental results
and COMSOL data analysis.\n
 - PLOT DISPERSION RELATION from COMSOL data: plot_dispersion(filename)
    (must be .csv and when exporting: choose parametric solutions > all > all > freq)\n
 - PLOT TRANSMISSION AMPLITUDE from COMSOL data: plot_amplitude(filename)
'''

import numpy as np
import matplotlib.pyplot as plt
from modal_match import T

c = 344820
h = 1.5
f1 = 5000
f2 = 30000


def read_data(filename):
    '''reads CSV file exported from COMSOL. returns data array and headers list
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    header = None
    data_start = 0
    for i, line in enumerate(lines):
        # comsol csv format comments begin with %
        if line.startswith('%'):
            header = line.strip('% \n')
            data_start = i + 1  # finds point where numeric data starts
        else:
            break

    # split header object into list of column names
    columns = [h.strip() for h in header.split(',')]

    # load numeric data from the start point
    data = np.loadtxt(filename, delimiter=',', skiprows=data_start)

    return data, columns


def plot_amplitude(filename, plot=True):
    '''plotting one angle slice from COMSOL csv exported data. For plot = 
    False, will just return a plot object which can be showed on one
    plot with other plotted data.'''
    data, columns = read_data(filename)

    data_num = data[:,2]

    plt.plot(data[:, 0], data_num, label='numerical', ls=':')
    if plot is True:
        plt.xlabel(columns[0])
        plt.ylabel('$|T|^2$')
        plt.legend()
        plt.title('Spaceplate - Transmission Amplitude')
        plt.grid()
        plt.show()


def plot1d_RT(filename):
    '''ONLY WORKS IF THIS IS NOT A PARAMETRIC SWEEP AND TABLE IS ONE COLUMN
    TRANSMISSION - THE OTHER REFLECTION.
    Comparing T and R. Input: filename = fpath'''
    data, columns = read_data(filename)

    plt.plot(data[:, 0], data[:,1], label='T')
    plt.plot(data[:, 0], data[:,2], label='R')

    plt.xlabel(columns[0])
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Spaceplate - Reflection vs Transmission')
    plt.grid()
    plt.show()


def plot_dispersion(filename):
    '''when exporting: choose parametric solutions > all > all > freq'''
    # reading in data
    with open(filename, 'r') as f:
        lines = f.readlines()

    # find the last header line (should start with % if from COMSOL, cant
    # imagine that will ever change?)
    for i, line in enumerate(lines):
        if line.startswith('%'):
            header = line.strip('% \n')
            data_start = i + 1
        else:
            break

    # extract frequency values from header
    header_parts = header.split(',')
    freqs = []
    for part in header_parts[1:]:
        if 'freq=' in part:
            freq = float(part.split('freq=')[1].split(' ')[0])
            freqs.append(freq/1000)

    freqs = np.array(freqs)  # y axis

    data = np.loadtxt(filename, delimiter=',', skiprows=data_start)
    Z = data[:, 1:]      # shape: (n_theta, n_freq)

    # for highlighting features of the plot
    Z = np.clip(Z, 0, 1)

    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    theta0 = data[:, 0]

    k0 = 2*np.pi*freqs*1000/c
    K0, THETA0 = np.meshgrid(k0, theta0)
    KX = K0*np.sin(np.deg2rad(THETA0))

    # plotting
    plt.pcolormesh(KX.T, freqs, Z.T, shading='auto', cmap='plasma')

    plt.xlabel(r'$k_x$ (1/mm)')
    plt.ylabel('Frequency (kHz)')
    plt.title('Spaceplate Dispersion Relation - COMSOL - Thermo Acoustics')
    plt.colorbar(label='Transmission')
    plt.xlim(right=0.47)
    plt.show()


def plot_comparison(filename, expfile, backfile):
    '''to plot comparison between theoretical, numerical and exeprimental data.
    comment out according to what you need to compare.'''

    ########################### analytical plotting ###########################
    N = 2000
    # frequency range
    k = 2*np.pi/c * np.linspace(f1, f2, N)

    # list of modes
    size = 3
    m = [(m1, m2) for m1 in range(-size, size+1) for m2 in range(-size, size+1)]
    m = list(dict.fromkeys(m))

    # get normalised transmission values
    Z = T(0, k, m)
    Z = np.abs(Z)**2
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    plt.plot(np.linspace(f1, f2, N), Z, label='analytical')

    ########################### numerical plotting ############################
    # for single angle sweep over frequency csv
    # plot_amplitude(filename, plot=False)
    
    # ==========================================================
    ############# for parametric sweep data plotting ########### 
    # ==========================================================
    # data, columns = read_data(filename)
    
    # freqs = []
    # for part in columns[1:]:
    #     if 'freq=' in part:
    #         freq = float(part.split('freq=')[1].split(' ')[0])
    #         freqs.append(freq)

    # eval = data[0]
    # plt.plot(freqs, eval[:-1], label='numerical', ls='--')  # for parametric sweep data
    # ==========================================================

    ########################### experimental plotting #########################
    from data_processing import plot1Damp
    freq, amp = plot1Damp(expfile, backfile)

    plt.scatter(freq, amp, label='experimental', s=7, color='r')

    ############################### plot all ##################################
    plt.title(f'Amplitude of Transmission from Spaceplate')
    plt.xlabel('Frequency (kHz)')
    # plt.xlim((f1, f2))
    plt.ylabel('$|T|^2$')
    plt.grid()
    plt.legend()
    plt.show()


def TAvsPA(TAfile, PAfile):
    # angle
    theta_idx = 40

    # thermoacoustic losses
    t_data, t_cols = read_data(TAfile)

    t_freqs = []
    for part in t_cols[1:]:
        if 'freq=' in part:
            freq = float(part.split('freq=')[1].split(' ')[0])
            t_freqs.append(freq/1000)
   
    t_eval = t_data[theta_idx]

    # pressure acoustics only
    p_data, p_cols = read_data(PAfile)
    
    p_freqs = []
    for part in p_cols[1:]:
        if 'freq=' in part:
            freq = float(part.split('freq=')[1].split(' ')[0])
            p_freqs.append(freq/1000)
   
    p_eval = p_data[theta_idx]
    theta = p_eval[0]

    if len(p_eval != t_eval):
        print('Data sets not same length')

    plt.plot(p_freqs, p_eval[1:], color='blue', label='PA')
    plt.plot(t_freqs, t_eval[1:], color='red', label='TA')

    plt.title(f'Thermo vs Pressure Acoustics. Theta = {theta}')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('$|T|^2$')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # filename_PA = "comsol_data/sp_df3dPA.csv"
    # filename_TA = "comsol_data/sp_df3dTA.csv"

    filename_PA = "comsol_data/sp_finalgeom_PA.csv"
    filename_TA = "comsol_data/sp_finalgeom_TA.csv"

    filename_TA = "ta_newgeom_t5again.csv"

    # filename_PA = 'comsol_data/sp_final_PA0.csv'

    filename = 'comsol_data/sp_finalgeom_0degPA.csv'

    expfile = "scan_data/scan_sp21.npy"
    backfile = "scan_data/scan_ns21.npy"

    filename2 = "sp_finalgeom_t5TA.csv"
    filename3 = 'Ta-stillbad0.csv'
    filename3 = 'Untitled2.csv'

    # ************ plot comparison lossless/ w. losses **************
    # TAvsPA(filename_TA, filename_PA)  

    # ***************** plot dispersion relation ********************
    plot_dispersion(filename_TA)

    # ***************** plot transmission normal ********************
    # plot1d(filename)

    # ************** plot transmission comparision ******************
    # plot_comparison(filename_PA, expfile, backfile)
