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
    base_ = UnitSquareMesh(para.Mbase,para.Mbase)
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
    
    u0 = interpolate(as_vector((-sin(pi*y)*cos(pi*x), cos(pi*y)*sin(pi*x))),Z.sub(0))
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
                             "ksp_atol": 1e-6,
                             "ksp_rtol": 1e-6,
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
                             'mat_type': 'aij',
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 100,
                             "ksp_gmres_restart": 100,
                             "ksp_atol": 1e-6,
                             "ksp_rtol": 1e-6,
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
    nnz = int(A.getInfo()['nz_allocated'])
    
    #Compute error
    u_exact = Function(Z.sub(0))
    val = as_vector((-cos(pi*x)*sin(pi*y)*exp(-2*pi**2*t),
                     sin(pi*x)*cos(pi*y)*exp(-2*pi**2*t)))
    u_exact.interpolate(val)
    
    l2err = assemble((u-u_exact)**2*dx)**0.5
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



def order_points(mesh_dm, points, ordering_type, prefix):
    '''Order a the points (topological entities) of a patch based on the adjacency graph of the mesh.
    :arg mesh_dm: the `mesh.topology_dm`
    :arg points: array with point indices forming the patch
    :arg ordering_type: a `PETSc.Mat.OrderingType`
    :arg prefix: the prefix associated with additional ordering options
    :returns: the permuted array of points                                                                        
    '''
    if ordering_type == "natural":
        return points
    subgraph = [numpy.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1] for p in points]
    ia = numpy.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
    ja = numpy.concatenate(subgraph).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, numpy.ones(ja.shape, PETSc.RealType)), comm=PETSc.COMM_SELF)
    A.setOptionsPrefix(prefix)
    rperm, _ = A.getOrdering(ordering_type)
    A.destroy()
    return points[rperm.getIndices()]


class ASMVankaStarPC(ASMPatchPC):
    '''Patch-based PC using closure of star of mesh entities implemented as an
    :class:`ASMPatchPC`.
    ASMVankaStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the closure of the star of the mesh entity
    specified by `pc_vanka_construct_dim` (or codim).
    This version includes the star of the "exclude_subspaces" in the patch
    '''

    _prefix = "pc_vankastar_"

    def get_patches(self, V):
        mesh = V._mesh
        mesh_dm = mesh.topology_dm
        if mesh.layers:
            warning("applying ASMVankaPC on an extruded mesh")

        # Obtain the topological entities to use to construct the stars
        depth = PETSc.Options().getInt(self.prefix + "construct_dim", default=-1)
        height = PETSc.Options().getInt(self.prefix + "construct_codim", default=-1)
        if (depth == -1 and height == -1) or (depth != -1 and height != -1):
            raise ValueError(f"Must set exactly one of {self.prefix}construct_dim or {self.prefix}construct_codim")

        exclude_subspaces = [int(subspace) for subspace in PETSc.Options().getString(self.prefix+"exclude_subspaces", default="-1").split(",")]
        ordering = PETSc.Options().getString(self.prefix+"mat_ordering_type", default="natural")
        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = []
        for (i, W) in enumerate(V):
            V_local_ises_indices.append(V.dof_dset.local_ises[i].indices)

        # Build index sets for the patches
        ises = []
        if depth != -1:
            (start, end) = mesh_dm.getDepthStratum(depth)
        else:
            (start, end) = mesh_dm.getHeightStratum(height)

        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                continue

            # Create point list from mesh DM
            star, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
            pt_array_star = order_points(mesh_dm, star, ordering, self.prefix)
            
            pt_array_vanka = set()
            for pt in star.tolist():
                closure, _ = mesh_dm.getTransitiveClosure(pt, useCone=True)
                pt_array_vanka.update(closure.tolist())

            pt_array_vanka = order_points(mesh_dm, pt_array_vanka, ordering, self.prefix)
            # Get DoF indices for patch
            indices = []
            for (i, W) in enumerate(V):
                section = W.dm.getDefaultSection()
                if i in exclude_subspaces:
                    loop_list = pt_array_star
                else:
                    loop_list = pt_array_vanka
                for p in loop_list:
                    dof = section.getDof(p)
                    if dof <= 0:
                        continue
                    off = section.getOffset(p)
                    # Local indices within W
                    W_indices = slice(off*W.value_size, W.value_size * (off + dof))
                    indices.extend(V_local_ises_indices[i][W_indices])
            iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            ises.append(iset)

        return ises



if __name__=="__main__":
    print(chorin(parameters()))
