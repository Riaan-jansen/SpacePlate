import numpy as np
import matplotlib.pyplot as plt
from modal_match import T, T_DF, T3, T2, T4
from mm_dispersion import T_dispersion

c = 343000
h = 1.6
f1 = 10000
f2 = 40000

filename = "comsol_data/spaceplate_data_RT3d.csv"
filename_df = "comsol_data/spaceplate_df3dTA.csv"
filename2 = "comsol_data/spaceplate_dspn4.csv"


def read_data(filename):
    '''reads file. returns data array and headers list.'''
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


def plot1d(filename, DF=False):
    data, columns = read_data(filename)
    N = len(data[:,0])
    N = 400
    k = 2*np.pi/c * np.linspace(f1, f2, N)
    m = [(0,0)]
    size = 2

    m = [(m1, m2) for m1 in range(-size, size+1) for m2 in range(-size, size+1)]
    m = list(dict.fromkeys(m))
    if DF is True:
        Z = T_DF(k, m, h)
    else:
        Z = T4(k, m, h)
    Z = np.abs(Z)**2
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    data_num = data[:,1]
    # data_num = data_num[::-1]

    plt.plot(data[:, 0], data_num, label='numerical')

    plt.plot(np.linspace(f1, f2, N), Z, label='analytical')

    # plt.yscale('log', base=2)
    plt.xlabel(columns[0])
    plt.ylabel('Transmission Value')
    plt.legend()
    plt.title('Spaceplate Data - 3d comsol geom')
    plt.grid()
    plt.show()


def plot1d_RT(filename):
    '''Comparing T and R. Input: filename = fpath'''
    data, columns = read_data(filename)

    plt.plot(data[:, 0], data[:,1], label='T')
    plt.plot(data[:, 0], data[:,2], label='R')
    plt.yscale('log', base=2)
    plt.xlabel(columns[0])
    plt.ylabel('Transmission Value')
    plt.legend()
    plt.title('Spaceplate Data - 3d comsol geom')
    plt.grid()
    plt.show()


def plot_data_2d(filename):
    '''choose parametric solutions > all > all > freq'''
    # Read header and data
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the last header line (starts with %)
    for i, line in enumerate(lines):
        if line.startswith('%'):
            header = line.strip('% \n')
            data_start = i + 1
        else:
            break  # Stop at first non-header line

    # Extract frequency values from header
    header_parts = header.split(',')
    freqs = []
    for part in header_parts[1:]:
        if 'freq=' in part:
            freq = float(part.split('freq=')[1].split(' ')[0])
            freqs.append(freq/1000)

    freqs = np.array(freqs)  # y axis

    data = np.loadtxt(filename, delimiter=',', skiprows=data_start)
    Z = data[:, 1:]      # shape: (n_theta, n_freq)

    Z = np.clip(Z, 0, 1)

    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    theta0 = data[:, 0]

    k0 = 2*np.pi*freqs*1000/c

    K0, THETA0 = np.meshgrid(k0, theta0)

    KX = K0*np.sin(np.deg2rad(THETA0))

    # Plot colourmap
    plt.pcolormesh(KX.T, freqs, Z.T, shading='auto', cmap='plasma')
    #extent = [KX.min(), KX.max(), freqs2.min(), freqs2.max()]
    #plt.imshow(Z.T, aspect='auto', origin='lower', extent=extent, cmap='jet')
    plt.xlabel(r'$k_x$ (1/mm)')
    plt.ylabel('Frequency (kHz)')
    plt.title('Spaceplate Dispersion Relation - COMSOL - Thermo Acoustics')
    plt.colorbar(label='Transmission')
    plt.xlim(right=0.47)
    plt.show()

    # scatter plot
    # plt.scatter(KX, K0, s=1, c=Z, cmap='jet')
    # plt.colorbar(label='Transmission coeff')
    # plt.title('Spaceplate dispersion relation: COMSOL')

    # plt.xlabel('$k_x$ (1/mm)')
    # plt.ylabel('Frequency (kHz)')

    # plt.show()

def plot_fine_sweep(filename2):
    # Read header and data
    with open(filename2, 'r') as f:
        lines = f.readlines()

    # Find the last header line (starts with %)
    for i, line in enumerate(lines):
        if line.startswith('%'):
            header = line.strip('% \n')
            data_start = i + 1
        else:
            break  # Stop at first non-header line

    # Extract frequency values from header
    header_parts = header.split(',')
    freqs = []
    for part in header_parts[1:]:
        if 'freq=' in part:
            freq = float(part.split('freq=')[1].split(' ')[0])
            freqs.append(freq)

    freqs = np.array(freqs)  # y axis

    data = np.loadtxt(filename2, delimiter=',', skiprows=data_start)

    N = 400
    k = 2*np.pi/c * np.linspace(f1, f2, N)
    m = [(0,0)]
    size = 3
    # m_list = [(x,x) for x in range(size)] + [(0,x) for x in range(size)] + [(x,0) for x in range(size)]
    m = [(m1, m2) for m1 in range(-size, size+1) for m2 in range(-size, size+1)]
    m = list(dict.fromkeys(m))

    Z = T_DF(k, m, h)
    Z = np.abs(Z)

    data_num = data[1,:]
    # data_num = data_num[::-1]

    plt.plot(freqs, data_num[1:], label='numerical')
    # plt.plot(data[:,0], Z, label='analytical')
    plt.plot(np.linspace(f1, f2, N), Z, label='analytical')
    # plt.plot(data[:, 0], data[:, 2], label=columns[2])
    plt.yscale('log', base=2)
    plt.xlabel('frequency')
    plt.ylabel('Transmission Value')
    plt.legend()
    plt.title('Spaceplate Data')
    plt.grid()
    plt.show()

def plot_comparison(filename):

    p_data, p_cols = read_data(filename)
        
    p_freqs = []
    for part in p_cols[1:]:
        if 'freq=' in part:
            freq = float(part.split('freq=')[1].split(' ')[0])
            p_freqs.append(freq/1000)

    p_eval = p_data[0]

    N = len(p_freqs)
    k = 2*np.pi/c * np.linspace(f1, f2, N)
    size = 2

    m = [(m1, m2) for m1 in range(-size, size+1) for m2 in range(-size, size+1)]
    m = list(dict.fromkeys(m))

    Z = T_dispersion(0, k, m)
    Z = np.abs(Z)**2
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    plt.plot(np.linspace(f1, f2, N)/1000, Z, label='analytical')
    plt.plot(p_freqs, p_eval[1:], label='numerical', ls='--')

    plt.title(f'Analytical vs Numerical Model')
    plt.xlabel('Frequency (kHz)')
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


# filename_PA = "comsol_data/sp_df3dPA.csv"
# filename_TA = "comsol_data/sp_df3dTA.csv"

filename_PA = "sp_finalgeom_PA.csv"
filename_TA = "sp_finalgeom_TA.csv"

#filename_df = "comsol_data/spaceplate_dspn4.csv"

# TAvsPA(filename_TA, filename_PA)

# plot_data_2d(filename_TA)
# plot_data_2d(filename2)
# import mm_dispersion
# plot1d(filename)
# plot1d(filename_df, DF=True)
plot_comparison(filename_PA)