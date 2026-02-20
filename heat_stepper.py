from firedrake import *
from irksome import DiscontinuousGalerkinTimeStepper, Dt, MeshConstant
from time import time

#Problem parameters used if running this script
class parameters:
    def __init__(self):
        self.N = 50
        self.dt = 0.001 #Specified instead of end time
        self.Mbase = 5
        self.Mref = 2
        self.degree = {'space': 1,
                       'time': 1}
        self.solver = None


#Solve the heat equation with timings
def heat(para=parameters):
    start = time()

    #Define mesh
    distribution_parameters={"partition": True,
                             "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    base_ = UnitSquareMesh(para.Mbase,para.Mbase,
                           distribution_parameters=distribution_parameters)
    spatial_mh = MeshHierarchy(base_,para.Mref)
    mesh = spatial_mh[-1]
   
    #Define function space
    space_element = FiniteElement("CG", triangle, para.degree['space'])
    U = FunctionSpace(mesh,space_element)

    #Define initial condition
    x, y = SpatialCoordinate(mesh)
    u = Function(U).interpolate(sin(pi*x)+cos(2*pi*y))

    MC = MeshConstant(mesh)
    dt = MC.Constant(para.dt)
    t = MC.Constant(0.0)

    #Set up residual and BC
    phi = TestFunction(U)

    F = (inner(Dt(u),phi) + inner(grad(u),grad(phi))) * dx(degree=16)

    #Set up solver

    if para.solver=='lu':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             'pc_type': 'lu'}
    else:
        solver_parameters = {"mat_type": "aij",
                             "snes_type": "ksponly",
                             "ksp_type": "fgmres",
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 100,
                             "ksp_gmres_restart": 100,
                             "ksp_atol": 1e-14,
                             "ksp_rtol": 1e-10,
                             "pc_type": "mg",
                             "mg_levels": {
                                 "ksp_type": "chebyshev",
                                 "ksp_chebyshev_esteig": "0,0.25,0,1.05",
                                 "ksp_max_it": 2,
                                 "ksp_convergence_test": "skip",
                                 "pc_type": "python",
                                 "pc_python_type": "firedrake.PatchPC",
                                 "patch": {
                                     "pc_patch": {
                                         "save_operators": True,
                                         "partition_of_unity": False,
                                         "construct_type": "star",
                                         "construct_dim": 0,
                                         "sub_mat_type": "seqdense",
                                         "dense_inverse": True,
                                         "precompute_element_tensors": None},
                                     "sub": {
                                         "ksp_type": "preonly",
                                         "pc_type": "lu"}}},
                             "mg_coarse": {
                                 "pc_type": "lu",
                                 "pc_factor_mat_solver_type": "mumps"}
                             }

    stepper = DiscontinuousGalerkinTimeStepper(F, para.degree['time'], t, dt, u, bcs=None,
                                               solver_parameters=solver_parameters)


    start_solve = time()
    count = 0
    while (count < para.N):
        stepper.advance()
        print(float(t), flush=True)
        t.assign(float(t) + float(dt))
        count += 1

    end = time()
    # After the solve, we can retrieve some statistics about the solver::

    steps, nonlinear_its, linear_its = stepper.solver_stats()

    #Get number of nonzero entries
    A, P = stepper.solver.snes.ksp.getOperators()
    nnz = int(A.getInfo()['nz_used'])

    print("Total number of timesteps was %d" % (steps))
    print("Average number of nonlinear iterations per timestep was %.2f" % (nonlinear_its/steps))
    print("Average number of linear iterations per timestep was %.2f" % (linear_its/steps))

    #Output relevant info
    out = {'dof': U.dim(),
           'nnz': nnz,
           'iterations': linear_its/steps,
           'time_total': end-start,
           'time_solve': end-start_solve}

    return out


if __name__=="__main__":
    print(heat(parameters()))
