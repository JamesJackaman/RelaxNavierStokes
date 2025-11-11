'''
Generate plots for cost model
'''
import numpy as np
#import mpl_helper
import matplotlib.pylab as plt
import matplotlib
import mpl_helper
from matplotlib.ticker import AutoMinorLocator, FixedLocator

#Set up the model parameters
Vcycles = 12
time_step_Vcycles = 3

cycle_complexity = (2+2)*(4/3)
num_relaxations = 4

Mref = 4
if Mref==2:
    num_patches_ = (41**2)
    M_ = 41
elif Mref==3:
    num_patches_ = (81**2)
    M_ = 81
elif Mref==4:
    num_patches_ = (161**2)
    M_ = 161
else:
    raise NotImplementedError('Invalid choice of Mref')
num_temporal = 1000

alpha = 1.5e-3
beta = 1.5e-6


def spatial_dofs(order):
    vel = 7 + 12*(order-1) + 6*(order-1)*(order-2)/2
    pres = 1 + 6*(order-1-1) + 6*(order-1-1)*(order-2-1)/2
    return 2*vel+pres

def spatial_dofs_core(order,n,cores=8):
    def heat_dofs(order,n):
        return ((n+1)**2 + (3*n**2 + 2*n)*(order-1) + (2*n**2)*((order-1)*(order-2)/2))
        
    return (2*heat_dofs(order,n) + heat_dofs(order-1,n) ) / cores

### The following assume order is the velocity order, and it is at least 2

def velocity_dofs_vertex(order):
    return 2

def pressure_dofs_vertex(order):
    return 1

def velocity_dofs_edge(order):
    return 2*(order-1)

def pressure_dofs_edge(order):
    return order-2

def velocity_dofs_element(order):
    return 2*(order-1)*(order-2)/2

def pressure_dofs_element(order):
    return (order-2)*(order-3)/2

def velocity_velocity_nonzeros(order):
    central_vertex = velocity_dofs_vertex(order) * (7*velocity_dofs_vertex(order) + 12*velocity_dofs_edge(order) + 6*velocity_dofs_element(order))

    edge_vertex = 6*velocity_dofs_vertex(order) * (4*velocity_dofs_vertex(order) + 5*velocity_dofs_edge(order) + 2*velocity_dofs_element(order))

    interior_edge = 6*velocity_dofs_edge(order) * (4*velocity_dofs_vertex(order) + 5*velocity_dofs_edge(order) + 2*velocity_dofs_element(order))

    exterior_edge = 6*velocity_dofs_edge(order) * (3*velocity_dofs_vertex(order) + 3*velocity_dofs_edge(order) + 1*velocity_dofs_element(order))

    element = 6*velocity_dofs_element(order) * (3*velocity_dofs_vertex(order) + 3*velocity_dofs_edge(order) + 1*velocity_dofs_element(order))

    return central_vertex + edge_vertex + interior_edge + exterior_edge + element

def velocity_pressure_nonzeros(order):
    central_vertex = velocity_dofs_vertex(order) * (pressure_dofs_vertex(order) + 6*pressure_dofs_edge(order) + 6*pressure_dofs_element(order))
    
    edge_vertex = 6*velocity_dofs_vertex(order) * (pressure_dofs_vertex(order) + 3*pressure_dofs_edge(order) + 2*pressure_dofs_element(order))

    interior_edge = 6*velocity_dofs_edge(order) * (pressure_dofs_vertex(order) + 3*pressure_dofs_edge(order) + 2*pressure_dofs_element(order))

    exterior_edge = 6*velocity_dofs_edge(order) * (pressure_dofs_vertex(order) + 2*pressure_dofs_edge(order) + 1*pressure_dofs_element(order))

    element = 6*velocity_dofs_element(order) * (pressure_dofs_vertex(order) + 2*pressure_dofs_edge(order) + pressure_dofs_element(order))

    return central_vertex + edge_vertex + interior_edge + exterior_edge + element
        
def spatial_nonzeros(order):

    return velocity_velocity_nonzeros(order) + 2*velocity_pressure_nonzeros(order)
    
    
def temporal_dofs(order):
    return order+1

#define cost model parameters
args = [1.28626836e+00, 2.45362659e+00, 1.00128576e+00, 5.90048167e-04, 3.51936113e-07, 1.50706914e-03]
p1 = args[0]
p2 = args[1]
p3 = args[2]
c1 = args[3]
c2 = args[4]
c3 = args[5]

def generate_global(space_order,time_order,s_processors=8):

    num_patches = num_patches_/s_processors
    M = M_/(s_processors**(0.5))

    num_spatial_levels = np.log2(M_)


    cost = c1 * (spatial_dofs_core(space_order,np.sqrt(num_patches_),s_processors)*temporal_dofs(time_order))**p1 \
        + c2 * (time_step_Vcycles * cycle_complexity * num_patches * num_temporal) * (temporal_dofs(time_order))**p2 * spatial_nonzeros(space_order)**p3
    
    comm = 4 * (alpha * num_temporal * time_step_Vcycles * num_relaxations * num_spatial_levels
                + beta * num_temporal * time_step_Vcycles * num_relaxations * 2 * M * spatial_dofs(space_order) * temporal_dofs(time_order)
                )
    return cost + comm


def generate_cyclic(space_order,time_order,processors=1,s_processors=8):

    num_patches = num_patches_/s_processors
    M = M_/(s_processors**0.5)

    num_spatial_levels = np.log2(M_)

    cost = c1 * (spatial_dofs_core(space_order,np.sqrt(num_patches_),s_processors)*temporal_dofs(time_order))**p1 \
        + c2 * ( 3 * num_temporal/processors + processors) * (Vcycles * cycle_complexity * num_patches) * (temporal_dofs(time_order))**p2 * spatial_nonzeros(space_order)**p3

    comm = 4 * (alpha * Vcycles * num_relaxations * num_spatial_levels
                + alpha * 0.25 * Vcycles * num_relaxations * num_spatial_levels
                + beta * Vcycles * num_relaxations * (num_temporal / processors) * 2 * M * spatial_dofs(space_order) * temporal_dofs(time_order)
                + beta * Vcycles * num_relaxations * (8/3) * spatial_dofs(space_order) * temporal_dofs(time_order) * num_patches
                )

    return cost + comm




def FigureOptimise():
    #Initialise plots
    (fig, axes_array, cax) = mpl_helper.make_fig_array(
        n_vert=1,
        n_horiz=2,
        axis_ratio=1.5,
        figure_width=5.1,
        left_margin=0.65,
        right_margin=0.1,
        bottom_margin=1.2,
        top_margin=0.25,
        colorbar_location="bottom",
        colorbar="single",
        colorbar_width = 0.1,
        colorbar_offset=0.5,
        horiz_sep=0.4,
        vert_sep=0.1,
        share_x_axes=False,
        share_y_axes=True,
    )
    
    t_orders = [0,2]
    s_orders = [1,3]
    S_processors = [1,2,4,8,16,32,64,128,256,512]
    T_processors = [1,2,4,8,16,32,64,128,256,512]

    #Initialise first plot

    axes = axes_array[0][0] #First plot

    #Generate first plot data
    t_order = t_orders[0]
    s_order = s_orders[0]
    time_space_low = np.zeros_like(S_processors)
    time_cyclic_low = np.zeros((len(S_processors),len(S_processors)))

    for i in range(len(S_processors)):
        time_space_low[i] = generate_global(s_order,t_order,
                                     s_processors=S_processors[i])
        for j in range(len(T_processors)):
            time_cyclic_low[i,j] = generate_cyclic(s_order,t_order,
                                               processors=T_processors[j],
                                               s_processors=S_processors[i])

    #Generate second plot data
    t_order = t_orders[1]
    s_order = s_orders[1]
    time_space_high = np.zeros_like(S_processors)
    time_cyclic_high = np.zeros((len(S_processors),len(S_processors)))

    for i in range(len(S_processors)):
        time_space_high[i] = generate_global(s_order,t_order,
                                     s_processors=S_processors[i])
        for j in range(len(T_processors)):
            time_cyclic_high[i,j] = generate_cyclic(s_order,t_order,
                                               processors=T_processors[j],
                                               s_processors=S_processors[i])


    #Get colorbar range
    vmin = min(time_cyclic_low.min(),time_cyclic_high.min())
    vmax = max(time_cyclic_low.max(),time_cyclic_high.max())
    norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)

    l1 = axes.pcolor(S_processors,T_processors,time_cyclic_low,norm=norm)
    axes.set_xscale('log')
    axes.set_yscale('log')
    xy_labels = [r'$2^0$',r'$2^1$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$',r'$2^6$',r'$2^7$',r'$2^8$',r'$2^9$']
    axes.set_yticks(T_processors, xy_labels)
    axes.set_xticks(S_processors, xy_labels)

    axes.set_xlabel('Spatial cores')
    axes.set_ylabel('Temporal cores')

    axes.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)

    axes.set_title('Low order')

    #Second plot now
    axes = axes_array[0][1]
    

    l2 = axes.pcolor(S_processors,T_processors,time_cyclic_high,norm=norm)
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_yticks(T_processors, xy_labels)
    axes.set_xticks(S_processors, xy_labels)
    fig.colorbar(l2,cax=cax,orientation='horizontal',label='Time (s)')

    axes.set_title('High order')

    axes.set_xlabel('Spatial cores')

    #Remove unwanted axes
    mpl_helper.remove_fig_array_axes(
        axes_array,
        remove_x_axes=True,
        remove_y_axes=True,
    )

    axes.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)
    
    plt.savefig('VarySpaceTimeProcessors.pdf')

    plt.close()


def FigureFixedCores():
    #Initialise plots
    (fig, axes_array) = mpl_helper.make_fig_array(
        n_vert=1,
        n_horiz=2,
        axis_ratio=1.5,
        figure_width=5.1,
        left_margin=0.65,
        right_margin=0.1,
        bottom_margin=1.2,
        top_margin=0.25,
        horiz_sep=0.2,
        vert_sep=0.1,
        share_x_axes=True,
        share_y_axes=True,
    )

    t_orders = [0,2]
    s_orders = [1,3]

    #Initialise colors
    colors = ['tab:blue',
	      'tab:orange',
	      'tab:green',
	      'tab:red',
	      'tab:blue',
              'tab:purple',]
    
    #Generate first set of data
    t_order = t_orders[0]
    s_order = s_orders[0]
    
    cores_lo = np.array([4,8,16,32,64,128,256,512,1024,2048, 4096, 8192, 16384, 32768], dtype="float64")
    cores_8 = cores_lo[1:]/8
    cores_16 = cores_lo[2:]/16
    cores_32 = cores_lo[3:]/32
    cores_64 = cores_lo[4:]/64
    cores_128 = cores_lo[5:]/128
    cores_256 = cores_lo[6:]/256
    cores_512 = cores_lo[7:]/512
    cores_1024 = cores_lo[8:]/1024
    cores_2048 = cores_lo[9:]/2048
    cores_4096 = cores_lo[10:]/4096
    cores_16384 = cores_lo[12:]/16384

    timings = np.zeros_like(cores_lo)
    for i in range(len(timings)):
        timings[i] = generate_global(s_order,t_order,
                              s_processors=cores_lo[i])
    timings_8 = np.zeros_like(cores_8)
    for i in range(len(timings_8)):
        timings_8[i] = generate_cyclic(s_order,t_order,
                                       processors=cores_8[i],
                                       s_processors=8)
    timings_16 = np.zeros_like(cores_16)
    for i in range(len(timings_16)):
        timings_16[i] = generate_cyclic(s_order,t_order,
                                       processors=cores_16[i],
                                        s_processors=16)
    timings_32 = np.zeros_like(cores_32)
    for i in range(len(timings_32)):
        timings_32[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_32[i],
                                        s_processors=32)

    timings_64 = np.zeros_like(cores_64)
    for i in range(len(timings_64)):
        timings_64[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_64[i],
                                        s_processors=64)
    timings_128 = np.zeros_like(cores_128)
    for i in range(len(timings_128)):
        timings_128[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_128[i],
                                        s_processors=128)
    timings_256 = np.zeros_like(cores_256)
    for i in range(len(timings_256)):
        timings_256[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_256[i],
                                         s_processors=256)
    timings_1024 = np.zeros_like(cores_1024)
    for i in range(len(timings_1024)):
        timings_1024[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_1024[i],
                                          s_processors=1024)
    timings_4096 = np.zeros_like(cores_4096)
    for i in range(len(timings_4096)):
        timings_4096[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_4096[i],
                                          s_processors=4096)
    timings_16384 = np.zeros_like(cores_16384)
    for i in range(len(timings_16384)):
        timings_16384[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_16384[i],
                                          s_processors=16384)


    ms = 4
        
    axes = axes_array[0][0] #First plot

    l1 = axes.plot(cores_lo,timings, linestyle='solid',color=colors[0],marker='o',markersize=ms,label='Time stepping')[0]
    l2 = axes.plot(16*cores_16,timings_16, linestyle=(0, (1, 4)),color=colors[1],marker='x',markersize=ms,label='16 spatial cores')[0]
    l3 = axes.plot(64*cores_64,timings_64, linestyle=(0, (1, 3)),color=colors[2],marker='x',markersize=ms,label='64 spatial cores')[0]
    l4 = axes.plot(256*cores_256,timings_256, linestyle=(0, (1, 2)),color=colors[3],marker='x',markersize=ms,label='256 spatial cores')[0]
    l5 = axes.plot(1024*cores_1024,timings_1024, linestyle=(0, (1, 1)),color=colors[4],marker='x',markersize=ms,label='1024 spatial cores')[0]

    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.set_xlabel('Cores')
    axes.set_ylabel('Time (s)')
    axes.set_title('Low order')

    axes.set_ylim((1e0,1e5))

    axes = axes_array[0][1] #Second plot
    core_multiplier = 32
    cores = core_multiplier*np.array([4,8,16,32,64,128,256,512,1024,2048, 4096, 8192, 16384, 32768,65536], dtype="float64")
    cores_8 = cores[1:]/(8*core_multiplier)
    cores_16 = cores[2:]/(16*core_multiplier)
    cores_32 = cores[3:]/(32*core_multiplier)
    cores_64 = cores[4:]/(64*core_multiplier)
    cores_128 = cores[5:]/(128*core_multiplier)
    cores_256 = cores[6:]/(256*core_multiplier)
    cores_512 = cores[7:]/(512*core_multiplier)
    cores_1024 = cores[8:]/(1024*core_multiplier)
    cores_2048 = cores[9:]/(2048*core_multiplier)
    cores_4096 = cores[10:]/(4096*core_multiplier)
    cores_16384 = cores[12:]/(16384*core_multiplier)


    #Generate second set of data
    t_order = t_orders[1]
    s_order = s_orders[1]
    timings = np.zeros_like(cores)
    for i in range(len(timings)):
        timings[i] = generate_global(s_order,t_order,
                              s_processors=cores[i])
    timings_8 = np.zeros_like(cores_8)
    for i in range(len(timings_8)):
        timings_8[i] = generate_cyclic(s_order,t_order,
                                       processors=cores_8[i],
                                       s_processors=core_multiplier*8)
    timings_16 = np.zeros_like(cores_16)
    for i in range(len(timings_16)):
        timings_16[i] = generate_cyclic(s_order,t_order,
                                       processors=cores_16[i],
                                        s_processors=core_multiplier*16)
    timings_32 = np.zeros_like(cores_32)
    for i in range(len(timings_32)):
        timings_32[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_32[i],
                                        s_processors=core_multiplier*32)

    timings_64 = np.zeros_like(cores_64)
    for i in range(len(timings_64)):
        timings_64[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_64[i],
                                        s_processors=core_multiplier*64)
    timings_128 = np.zeros_like(cores_128)
    for i in range(len(timings_128)):
        timings_128[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_128[i],
                                        s_processors=core_multiplier*128)
    timings_256 = np.zeros_like(cores_256)
    for i in range(len(timings_256)):
        timings_256[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_256[i],
                                         s_processors=core_multiplier*256)
    timings_1024 = np.zeros_like(cores_1024)
    for i in range(len(timings_1024)):
        timings_1024[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_1024[i],
                                          s_processors=core_multiplier*1024)
    timings_4096 = np.zeros_like(cores_4096)
    for i in range(len(timings_4096)):
        timings_4096[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_4096[i],
                                          s_processors=core_multiplier*4096)
    timings_16384 = np.zeros_like(cores_16384)
    for i in range(len(timings_16384)):
        timings_16384[i] = generate_cyclic(s_order,t_order,
                                        processors=cores_16384[i],
                                          s_processors=core_multiplier*16384)


    l7a = axes.plot(cores,timings, linestyle='solid',color=colors[0],marker='o',markersize=ms,label='Time stepping')[0]
    l9 = axes.plot(core_multiplier*64*cores_64,timings_64, linestyle=(0, (1, 4)),color=colors[1],marker='*',markersize=ms,label='32*64 spatial cores')[0]
    l10 = axes.plot(core_multiplier*256*cores_256,timings_256, linestyle=(0, (1, 3)),color=colors[2],marker='*',markersize=ms,label='32*256 spatial cores')[0]
    l11 = axes.plot(core_multiplier*1024*cores_1024,timings_1024, linestyle=(0, (1, 2)),color=colors[3],marker='*',markersize=ms,label='32*1024 spatial cores')[0]
    l12 = axes.plot(core_multiplier*4096*cores_4096,timings_4096, linestyle=(0, (1, 1)),color=colors[4],marker='*',markersize=ms,label='32*4096 spatial cores')[0]


    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.set_xlabel('Cores')
    axes.set_title('High order')


    #Remove unwanted axes
    mpl_helper.remove_fig_array_axes(
        axes_array,
        remove_x_axes=True,
        remove_y_axes=True,
    )
        
    fig.legend((l1,
                l2,l3,l4,l5,
                l9,l10,l11,l12),
               ('Time stepping',
                '16 spatial cores',
                '64 spatial cores',
                '256 spatial cores',
                '1024 spatial cores',
                '2048 spatial cores',
                '8192 spatial cores',
                '32768 spatial cores',
                '131072 spatial cores'
                ),
               loc='lower left',
               ncol=3, bbox_to_anchor=(0.125,0.05))
    
    plt.savefig('FixedSpaceProcessors.pdf')
    plt.close()

    return None

if __name__=="__main__":    
    FigureOptimise()
    FigureFixedCores()
