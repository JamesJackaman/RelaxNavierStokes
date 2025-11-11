import numpy as np
import scipy.optimize as spo
import functools as ft

Vcycles = 3
cycle_complexity = (2+2)*(4/3)
num_patches42 = (81**2)/8
num_patches44 = (161**2)/8
num_temporal = 20

# Transcribing Table 4.2, as temporal degree, spatial degree, time
data42 = np.array([[0, 1, 7.0],
                   [1, 1, 14.0],
                   [2, 1, 23.1],
                   [3, 1, 35.3],
                   [0, 2, 17.9],
                   [1, 2, 48.4],
                   [2, 2, 113.4],
                   [3, 2, 187.0],
                   [0, 3, 48.9],
                   [1, 3, 183.1],
                   [2, 3, 424.4],
                   [3, 3, 784.6]])
size42 = 12

# Transcribing Table 4.4, as temporal degree, spatial degree, time
data44 = np.array([[0, 1, 21.3],
                   [1, 1, 50.8],
                   [2, 1, 86.3],
                   [3, 1, 132.2],
                   [0, 2, 148.7],
                   [1, 2, 375.5],
                   [2, 2, 633.6],
                   [3, 2, 1064.0],
                   [0, 3, 373.4],
                   [1, 3, 1090.0]])
size44 = 10

def spatial_dofs(order):
    return 1 + 6*(order-1) + 6*(order-1)*(order-2)/2

def spatial_dofs_core(order,n):
    return ((n+1)**2 + (3*n**2 + 2*n)*(order-1) + (2*n**2)*((order-1)*(order-2)/2)) / 8

def spatial_nonzeros(order):
    return spatial_dofs(order) + 6*(order-1)*(order + 2*(order-1) + 2*(order-1)*(order-2)/2) + 6*((order-1)*(order-2)/2)*((order+1)*(order)/2)

def temporal_dofs(order):
    return order+1

def combine_functions(func_list):
    def combined_function(*args, **kwargs):
        return [f(*args, **kwargs) for f in func_list]
    return combined_function

for i in np.arange(1,6):
    print(f"{i}, DoFs {spatial_dofs(i)}, NNZ {spatial_nonzeros(i)}, NNZ/row {spatial_nonzeros(i)/spatial_dofs(i)}")

A = np.zeros((size42+size44, 4))
b = np.zeros((size42+size44, 1))

func_list = []

def cost(args,k=1,q=0,val=0,n=1,num_patches=-1):
    p1 = args[0]
    p2 = args[1]
    p3 = args[2]
    c1 = args[3]
    c2 = args[4]
    c3 = args[5]
    
    out = c1 * (spatial_dofs_core(k,n)*temporal_dofs(q))**p1 \
        + c2 * (Vcycles * cycle_complexity * num_patches * num_temporal) * (temporal_dofs(q))**p2 * spatial_nonzeros(k)**p3 + c3
    return out - val

for i in np.arange(size42):
    A[i, 0] = 1
    A[i, 1] = temporal_dofs(data42[i, 0])
    A[i, 2] = spatial_dofs(data42[i, 1])
    A[i, 3] = spatial_nonzeros(data42[i, 1])
    #b[i] = np.log(data42[i,2] / (Vcycles * cycle_complexity * num_patches42 * num_temporal))

    print('k = ', data42[i,1])
    
    func_list.append(ft.partial(cost,k=data42[i, 1], q=data42[i,0], val=data42[i,2], num_patches=num_patches42, n=80))

for i in np.arange(size44):
    A[size42+i, 0] = 1
    A[size42+i, 1] = np.log(temporal_dofs(data44[i, 0])) + np.log(spatial_nonzeros(data44[i, 1]))
    A[size42+i, 2] = np.log(spatial_dofs(data44[i, 1]))
    A[size42+i, 3] = np.log(spatial_nonzeros(data44[i, 1]))
    #b[size42+i] = np.log(data44[i,2] / (Vcycles * cycle_complexity * num_patches44 * num_temporal))

    func_list.append(ft.partial(cost,k=data44[i, 1], q=data44[i,0], val=data44[i,2], num_patches=num_patches44, n=160))


objective = combine_functions(func_list)

lower_bounds = [1, 1, 1, 0, 0, 0]
upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

result = spo.least_squares(objective, [1,1,1,1,1,1], bounds=(lower_bounds,upper_bounds))

print(result.x)

x, _, _, _ = np.linalg.lstsq(A[:,[0, 1]],b)
x = np.append(x, [x[1]])
print(f"Temporal power is {x[1]}")
#print(f"Patch size power is {x[2]}")
print(f"Patch nonzeros power is {x[2]}")
print(f"Implied constant is {np.exp(x[0])}")

const = np.exp(x[0])

print("For Mref = 3 data:")
for i in np.arange(size42):
    model = cost(result.x,k=data42[i,1],q=data42[i,0],val = 0, num_patches=num_patches42, n=80)
    print(f"For temporal order {data42[i,0]} and spatial order {data42[i,1]}, true cost is {data42[i,2]} and model cost is {model}, relative error is {100*np.abs(data42[i,2]-model)/data42[i,2]}%")

print("For Mref = 4 data:")
for i in np.arange(size44):
    model = cost(result.x,k=data44[i,1],q=data44[i,0],val = 0, num_patches=num_patches44, n=160)
    print(f"For temporal order {data44[i,0]} and spatialorder {data44[i,1]}, true cost is {data44[i,2]} and model cost is {model}, relative error is {100*np.abs(data44[i,2]-model)/data44[i,2]}%")
