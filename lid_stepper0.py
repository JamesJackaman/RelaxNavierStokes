"""
Lid driven cavity
"""
#Global imports
from firedrake import *
import numpy as np
from time import time

#Parallel safe printing
Print = PETSc.Sys.Print

class parameters:
    def __init__(self):
        self.N = 20
        self.dt = 0.001 #Specified instead of end time
        self.Mbase = 10
        self.Mref = 3
        self.degree = {'space': 2}
        self.R = Constant(100) #Reynolds number
        self.alpha = Constant(1) #Diffusion constant
        self.plot = True
        self.solver = 'one_step'
    

def lid(para=parameters):
    
    start = time()
    
    def plus(v):
        return -0.5*jump(v,n[2]) + avg(v)
    
    #Define mesh
    distribution_parameters={"partition": True,
                             "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    base_ = UnitSquareMesh(para.Mbase,para.Mbase,
                           distribution_parameters=distribution_parameters)
    spatial_mh = MeshHierarchy(base_,para.Mref)

    mesh = spatial_mh[-1]

    n = FacetNormal(mesh)
    
    #Define function space
    CG2 = VectorFunctionSpace(mesh,"CG",para.degree['space']+1,dim=2)
    CG1 = FunctionSpace(mesh,"CG",para.degree['space'])
    Z = MixedFunctionSpace((CG2,CG1))
    
    #Define initial condition
    x, y = SpatialCoordinate(Z.mesh())

    z0 = Function(Z)
    u0, p0 = z0.subfunctions
    
    #Set up residual
    z = Function(Z)
    u, p = split(z)
    phi, psi = TestFunctions(Z)

    gradu = as_vector([u.dx(0),
                       u.dx(1)])

    gradphi = as_vector([phi.dx(0),
                         phi.dx(1)])
    
    F_1 = para.R * inner(dot(u, gradu),phi) * dx \
        + p * (phi[0].dx(0) + phi[1].dx(1)) * dx \
        + para.alpha * inner(gradu,gradphi) * dx

    F_2 = (u[0].dx(0) + u[1].dx(1)) * psi * dx

    F_time = para.dt**(-1)*inner(u - u0, phi) * dx

    F = F_1 + F_2 + F_time
    F = para.dt * F #rescale

    class PressureFixBC(DirichletBC):
        def __init__(self, V, val, subdomain):
            super().__init__(V, val, subdomain)
            sec = V.dm.getDefaultSection()
            dm = V.mesh().topology_dm
            coordsSection = dm.getCoordinateSection()
            coordsDM = dm.getCoordinateDM()
            dim = dm.getCoordinateDim()
            coordsVec = dm.getCoordinatesLocal()
            (vStart, vEnd) = dm.getDepthStratum(0)
            indices = []
            for pt in range(vStart, vEnd):
                x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
                if (x[1]==0.) and (x[0]==0.):
                    if dm.getLabelValue("pyop2_ghost", pt) == -1:
                        indices.append(pt)
            nodes = []
            for i in indices:
                num_of_layers = sec.getDof(i)
                if num_of_layers > 0:
                    off = sec.getOffset(i)
                    nodes.append(np.arange(off,off+num_of_layers))

            self.nodes = np.asarray(nodes, dtype=IntType)

            Print("Fixing nodes %s" % self.nodes)
    
    bc1 = DirichletBC(Z.sub(0),as_vector((0,0)),(1,2,3))
    bc2 = DirichletBC(Z.sub(0),as_vector((1,0)),4)
    bc3 = PressureFixBC(Z.sub(1),Constant(0),1)
    
    
    #Set up solver
    tol = 1e-8 #solver tolerance
    if para.solver=='lu':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             "pc_factor_shift_type":"nonzero",
                             'pc_type': 'lu',
                             'snes_max_it': 1000,
                             'snes_rtol': tol,
                             'snes_stol': tol,
                             'snes_monitor': None}
    elif para.solver=='one_step':    
        solver_parameters = {'snes_type': 'ksponly',
                             'mat_type': 'aij',
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 100,
                             "ksp_gmres_restart": 100,
                             # 'ksp_atol': tol,
                             # 'ksp_rtol': tol,
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
                             "mg_levels_pc_vankastar_exclude_subspaces": "1,2",
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
                             'mat_type': 'aij',
                             # 'ksp_atol': tol,
                             # 'ksp_rtol': tol,
                             'snes_atol': tol,
                             'snes_rtol': tol,
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 100,
                             "ksp_gmres_restart": 100,
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
        
    problem = NonlinearVariationalProblem(F, z, bcs=[bc1,bc2,bc3])
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    t = 0 #initalise time
    #set up iteration counters
    iterations = []
    nl_iterations = []
    #Initialise plots
    if para.plot:
        zfile = File('plots/lid_stepper.pvd')
        u0, p0 = z0.subfunctions
        u0.rename('u','u')
        p0.rename('p','p')
        zfile.write(u0,p0, time=0)
        
    
    start_solve = time()
    while t<para.N * para.dt:
        #Solve
        solver.solve()
        z0.assign(z)
        t += para.dt

        #Save iteration counts
        iterations.append(solver.snes.getLinearSolveIterations())
        nl_iterations.append(solver.snes.getIterationNumber())

        #Save solution (if plotting)
        if para.plot:
            zfile.write(u0,p0, time=t)
            
        

    end = time()

    print('iterations', iterations)
    print(np.mean(iterations))
    print('nl iterations', nl_iterations)
    print(np.mean(nl_iterations))
    
        
    #Output relevant info
    out = {'dof': Z.dim(),
           'iterations': np.mean(iterations),
           'newton iterations': np.mean(nl_iterations),
           'time_total': end-start,
           'time_solve': end-start_solve}

    

    return out



if __name__=="__main__":
    Print(lid(parameters()))
