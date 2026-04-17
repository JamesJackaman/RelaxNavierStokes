"""
Test script solving Navier-Stokes using
standard space-time finite elements
"""
#Global imports
from firedrake import *
from firedrake.petsc import PETSc
import matplotlib.pylab as plt
from time import time

class parameters:
    def __init__(self):
        self.N = 10
        self.dt = 0.01 #Specified instead of end time
        self.Mbase = 10
        self.Mref = 2
        self.degree = {'space': 1,
                       'time': 2}
        self.R = Constant(1) #Reynolds number
        self.alpha = Constant(1) #Diffusion constant
        self.plot = True
        self.solver = 'one_step'
    

def chorin(para=parameters):

    start = time()
    
    def plus(v):
        return -0.5*jump(v,n[2]) + avg(v)
    
    #Set up mesh
    distribution_parameters={"partition": True,
                             "overlap_type": (DistributedMeshOverlapType.VERTEX, 3)}
    base_ = UnitSquareMesh(para.Mbase,para.Mbase,
                           distribution_parameters=distribution_parameters)
    spatial_mh = MeshHierarchy(base_,para.Mref)
    mh = ExtrudedMeshHierarchy(spatial_mh, para.N*para.dt,
                        base_layer = para.N,
                        refinement_ratio=1,
                        extrusion_type='uniform')
    mesh = mh[-1]
    
    n = FacetNormal(mesh)
    
    #Define function space
    space_element1 = FiniteElement("CG", triangle, para.degree['space']+1)
    space_element2 = FiniteElement("CG", triangle, para.degree['space'])
    time_element = FiniteElement("DG", interval, para.degree['time'])
    spacetime_element1 = TensorProductElement(space_element1,time_element)
    spacetime_element2 = TensorProductElement(space_element2,time_element)
    Z = MixedFunctionSpace((VectorFunctionSpace(mesh,spacetime_element1,dim=2),
                           FunctionSpace(mesh,spacetime_element2)))
    
    #Define initial condition
    x, y, t = SpatialCoordinate(Z.mesh())

    z0 = Function(Z)
    u0, p0 = z0.subfunctions
    
    u0 = Function(Z.sub(0)).interpolate(as_vector((-sin(pi*y)*cos(pi*x), cos(pi*y)*sin(pi*x))))
    ut = as_vector((-cos(pi*x)*sin(pi*y)*exp(-2*pi**2*t),
                    sin(pi*x)*cos(pi*y)*exp(-2*pi**2*t)))
    
    #Set up residual
    z = Function(Z)
    u, p = split(z)
    phi, psi = TestFunctions(Z)

    gradu = as_vector([u.dx(0),
                       u.dx(1)])

    gradphi = as_vector([phi.dx(0),
                         phi.dx(1)])
    upwinding = jump(u[0],n[2]) * plus(phi[0]) + jump(u[1],n[2]) * plus(phi[1])
    
    F_1 = para.R * inner(dot(u,gradu),phi) * dx \
        + p * (phi[0].dx(0) + phi[1].dx(1)) * dx \
        + para.alpha * inner(gradu,gradphi) * dx

    F_2 = (u[0].dx(0) + u[1].dx(1)) * psi * dx
    
    F_time = inner(u.dx(2), phi) * dx - upwinding * dS_h
    F_ic = 0.5 * inner((u-u0),phi)*ds_b

    F = F_1 + F_2 + F_time + F_ic

    bc = DirichletBC(Z.sub(0),ut,"on_boundary")
    
    #Set up solver
    tol = 1e-8 #solver tolerance
    if para.solver=='lu':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             'pc_type': 'lu',
                             'snes_monitor': None}
    elif para.solver=='lu_step':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             'pc_type': 'lu',
                             'snes_type': 'ksponly'}
    elif para.solver=='one_step':
        solver_parameters = {'snes_type': 'ksponly',
                             'mat_type': 'aij',
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 100,
                             "ksp_gmres_restart": 100,
                             "ksp_atol": tol,
                             "ksp_rtol": tol,
                             'snes_atol': tol,
                             'snes_rtol': tol,
                             'pc_type': 'mg',
                             "pc_mg_type": "multiplicative",
                             "pc_mg_cycles": "v",
                             "mg_levels_ksp_type": "chebyshev",
                             "mg_levels_ksp_chebyshev_esteig": "0,0.25,0,1.05",
                             "mg_levels_ksp_max_it": 2,
                             "mg_levels_ksp_convergence_test": "skip",
                             "mg_levels_pc_type": "python",
                             "mg_levels_pc_python_type": __name__ + ".ASMVankaStarPC",
                             "mg_levels_pc_vankastar_construct_dim": 0,
                             "mg_levels_pc_vankastar_exclude_subspaces": "1",
                             "mg_levels_pc_vankastar_sub_sub_pc_type": "lu",
                             "mg_levels_pc_vankastar_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_coarse_pc_type": "python",
                             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                             "mg_coarse_assembled_pc_type": "lu",
                             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                             }
    else:
        solver_parameters = {'snes_type': 'newtonls',
                             'snes_ksp_ew': None,
                             'snes_monitor': None,
                             'snes_linesearch_monitor': None,
                             'snes_converged_reason': None,
                             'mat_type': 'aij',
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 20,
                             "ksp_gmres_restart": 20,
                             "ksp_atol": tol*0.1,
                             # "ksp_rtol": tol,
                             'snes_stol': 0,
                             'snes_atol': tol,
                             'snes_rtol': tol,
                             'pc_type': 'mg',
                             "pc_mg_type": "multiplicative",
                             "pc_mg_cycles": "v",
                             "mg_levels_ksp_type": "gmres",
                             "mg_levels_ksp_chebyshev_esteig": "0,0.25,0,1.05",
                             "mg_levels_ksp_max_it": 3,
                             "mg_levels_ksp_convergence_test": "skip",
                             "mg_levels_pc_type": "python",
                             # "mg_levels_pc_python_type": __name__ + ".ASMVankaStarPC",
                             # "mg_levels_pc_vankastar_construct_dim": 0,
                             # "mg_levels_pc_vankastar_exclude_subspaces": "1",
                             # "mg_levels_pc_vankastar_sub_sub_pc_type": "lu",
                             # "mg_levels_pc_vankastar_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_levels_pc_python_type": "firedrake.ASMVankaPC",
                             "mg_levels_pc_vanka_include_type": "star",
                             "mg_levels_pc_vanka_construct_dim": 0,
                             "mg_levels_pc_vanka_exclude_subspaces": "1",
                             "mg_levels_pc_vanka_sub_sub_pc_type": "lu",
                             "mg_levels_pc_vanka_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_coarse_pc_type": "python",
                             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                             "mg_coarse_assembled_pc_type": "lu",
                             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                             "mg_coarse_assembled_pc_factor_shift_type": "nonzero",
                             }

    
    problem = NonlinearVariationalProblem(F, z, bcs=[bc])
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)


    start_solve = time()
    
    #Solve
    solver.solve()

    end = time()
    
    iterations = solver.snes.getLinearSolveIterations()
    print('iterations', iterations)

    nl_iterations = solver.snes.getIterationNumber()
    print('nl_iterations', nl_iterations)

    #Get number of nonzero entries
    A, P = solver.snes.ksp.getOperators()
    nnz = int(A.getInfo()['nz_used'])
    
    #Compute error
    u_exact = as_vector((-cos(pi*x)*sin(pi*y)*exp(-2*pi**2*t),
                     sin(pi*x)*cos(pi*y)*exp(-2*pi**2*t)))
    
    l2err = assemble((u-u_exact)**2*dx(degree=16))**0.5
    print('Error is ', l2err)
    
    #Plot
    if para.plot:
        ufile = File('plots/chorin.pvd')
        u, p = z.subfunctions
        ufile.write(u,p)

    #Output relevant info
    out = {'dof': Z.dim(),
           'nnz': nnz,
           'error': l2err,
           'iterations': iterations,
           'newton iterations': nl_iterations,
           'time_total': end-start,
           'time_solve': end-start_solve}
           

    return out


if __name__=="__main__":
    print(chorin(parameters()))
