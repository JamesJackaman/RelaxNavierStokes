"""
Test script solving Navier-Stokes using
standard space-time finite elements
"""
#Global imports
from firedrake import *
from firedrake.petsc import PETSc
from irksome import DiscontinuousGalerkinTimeStepper, Dt, MeshConstant
from pyop2.datatypes import IntType
import matplotlib.pylab as plt
from time import time
from pyop2.datatypes import IntType

#Parallel safe printing
Print = PETSc.Sys.Print

#Parallel safe printing
Print = PETSc.Sys.Print

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
        self.solver = None
    

def chorin_stepper(para=parameters):

    start = time()
    
    #Set up mesh
    distribution_parameters={"partition": True,
                             "overlap_type": (DistributedMeshOverlapType.VERTEX, 3)}
    base_ = UnitSquareMesh(para.Mbase,para.Mbase,
                           distribution_parameters=distribution_parameters)
    spatial_mh = MeshHierarchy(base_,para.Mref)
    mesh = spatial_mh[-1]
    
    n = FacetNormal(mesh)
    
    #Define function space
    space_element1 = FiniteElement("CG", triangle, para.degree['space']+1)
    space_element2 = FiniteElement("CG", triangle, para.degree['space'])
    Z = MixedFunctionSpace((VectorFunctionSpace(mesh,space_element1,dim=2),
                           FunctionSpace(mesh,space_element2)))
    
    #Define initial condition
    x, y = SpatialCoordinate(Z.mesh())

    z0 = Function(Z)
    z0.sub(0).interpolate(as_vector((-sin(pi*y)*cos(pi*x), cos(pi*y)*sin(pi*x))))

    #Set up parameters for time stepping
    MC = MeshConstant(mesh)
    dt = MC.Constant(para.dt)
    t = MC.Constant(0.0)

    ut = as_vector((-cos(pi*x)*sin(pi*y)*exp(-2*pi**2*t),
                    sin(pi*x)*cos(pi*y)*exp(-2*pi**2*t)))

    #BC hack
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

    
    #Set up residual
    z = Function(Z)
    u, p = split(z)
    phi, psi = TestFunctions(Z)
    z.assign(z0)
    
    F_1 = para.R * inner(dot(grad(u),u),phi) * dx \
        + p * (div(phi)) * dx \
        + para.alpha * inner(grad(u),grad(phi)) * dx

    F_2 = (div(u)) * psi * dx
    
    F_time = inner(Dt(u), phi) * dx

    F = F_1 + F_2 + F_time

    bc1 = DirichletBC(Z.sub(0),ut,"on_boundary")
    bc2 = PressureFixBC(Z.sub(1),Constant(0),1)
    bc = [bc1, bc2]

    #nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])
    nsp = [(1, VectorSpaceBasis(constant=True, comm=mesh.comm))]
    
    #Set up solver
    tol = 1e-6 #solver tolerance
    if para.solver=='lu':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             'pc_type': 'lu',
                             'snes_monitor': None}
    else:
        ind_pressure = ",".join([str(2*i+1) for i in range(para.degree['time']+1)])
        solver_parameters = {'snes_type': 'newtonls',
                             'snes_ksp_ew': None,
                             'snes_monitor': None,
                             'snes_linesearch_monitor': None,
                             'snes_converged_reason': None,
                             'mat_type': 'aij',
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 200,
                             "ksp_gmres_restart": 20,
                             "ksp_atol": tol*0.1,
                             'snes_stol': 0,
                             'snes_atol': tol,
                             'snes_rtol': tol,
                             'pc_type': 'mg',
                             "pc_mg_type": "multiplicative",
                             "pc_mg_cycles": "v",
                             "mg_levels": {
                                 "ksp_type": "chebyshev",
                                 "ksp_chebyshev_esteig": "0,0.25,0,1.05",
                                 "ksp_max_it": 3,
                                 "ksp_convergence_test": "skip",
                                 "pc_type": "python",
                                 "pc_python_type": "firedrake.ASMVankaPC",
                                 "pc": {
                                     "vanka": {
                                         "include_type": "star",
                                         "construct_dim": 0,
                                         "exclude_subspaces": ind_pressure},
                                         #"sub_mat_type": "umfpack"},
                                     "vanka_sub_sub": {
                                         "pc_factor_mat_solver_type": "mumps",
                                         "pc_factor_shift_type": "nonzero",
                                         "pc_type": "lu"}}},
                             "mg_coarse": {
                                 "ksp_type": "preonly",
                                 "pc_type": "lu",
                                 "pc_factor_shift_type": "nonzero",
                                 "pc_factor_mat_solver_type": "mumps",
                                 'pc_factor_mat_mumps_icntl_14': 200}
                             }


    stepper = DiscontinuousGalerkinTimeStepper(F, para.degree['time'], t, dt, z, bcs=bc,
                                               solver_parameters=solver_parameters, nullspace=nsp)

    print('Solver parameters', stepper.solver.parameters)

    #solve
    start_solve = time()
    count = 0
    while (count < para.N):
        stepper.advance()
        print(float(t), flush=True)
        t.assign(float(t) + float(dt))
        count+=1

    end = time()

    #Get solver stats
    steps, nl_iterations, iterations = stepper.solver_stats()

    print('newstats', stepper.solver_stats())

    #Get number of nonzero entries
    A, P = stepper.solver.snes.ksp.getOperators()
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
           'iterations': iterations/steps,
           'newton iterations': nl_iterations/steps,
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
        V_local_ises_indices = tuple(iset.indices for iset in V.dof_dset.local_ises)

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
            
            pt_array_vanka = []
            for pt in reversed(pt_array_star):
                closure, _ = mesh_dm.getTransitiveClosure(pt, useCone=True)
                pt_array_vanka.extend(closure)

            # Grab unique points with stable ordering           
            pt_array_vanka = list(reversed(dict.fromkeys(pt_array_vanka)))

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
                    W_indices = slice(off*W.block_size, W.block_size * (off + dof))
                    indices.extend(V_local_ises_indices[i][W_indices])
            iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            ises.append(iset)

        return ises



if __name__=="__main__":
    print(chorin_stepper(parameters()))
