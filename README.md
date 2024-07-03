# Space-time waveform relaxation for Navier-Stokes

This repository is a suppliment to the paper "Space-time waveform relaxation for Navier-Stokes", and contains the implementation required to reproduce the numerical experiments presented within.

## Installation

This repository depends on:

### [Firedrake](https://www.firedrakeproject.org/)

The Firedrake installation script can be downloaded [here](https://www.firedrakeproject.org/download.html). The latest version can be installed with

```bash
python3 firedrake-install
```

The Firedrake version used in the associated paper can be installed with the command

```bash
python3 firedrake-install --doi FOO
```

The Zenodo URL for Firedrake can be found **HERE**, and has DOI **FOO**.

### Pandas

Data is visualised (in `*_vis.py`) with pandas, which should be pip installed into the virtual environment with

```bash
pip3 install pandas==1.5.0
```

## Code structure

This code solves 3D finite element discretisations, where two of the dimensions are spatial and one is temporal. Both the heat equation and imcompressible Navier-Stokes without external forcing are solved. The codebase is split into three test problems. 

- `heat*.py`: Code used to solve the heat equation.

- `chorin*.py`: Code used to solve Navier-Stokes to approximate the exact solution 

  ```math
      \begin{equation}
        \begin{split}
          \vec{u}(t,\vec{x})
          & =
          \begin{pmatrix}
            - \cos{\pi x_1} \sin{\pi x_2} \\
            \sin{\pi x_1} \cos{\pi x_2}
          \end{pmatrix} e^{-2\pi^2 t} \\
       \phi(t,\vec{x})
        & =
        R \frac{\pi}{4} \left(
          \cos{2\pi x_1} + \cos{2\pi x_2}\right)
        e^{-4\pi^2 t}
        .
        \end{split}
      \end{equation}
  ```

- `lid*.py`: Code used to solve Navier-Stokes for lid-driven cavity test problem.

For more information on each test case please see the associated paper.

### Function breakdown

For each test problem has multiple python files associated with it, which we now briefly outline the functionality of.

#### `{problem}.py`

Runs the associated problem with the parameters fixed by the `parameters` class at the top of the file. By default the WRMG solver is used, although by changing `parameters.solver='lu'` a direct solver may also be used. By default solution plots will be generated and saved to the `plots/` subfolder and may be visualised in paraview.

#### `{problem}_caller.py`

Calls the associated problem with optional arguments which can be passed to the `parameters` class. These files are auxiliary but heavily relied on for benchmarking. Run `python3 {problem}_caller.py --help` to see available options here. Note a default run will use WRMG but direct solvers can be used by passing the flag `--lu`. 

#### {problem}_generator.py

Generates the timing benchmarks displayed in the paper using 8 MPI cores (by default). Runs can also be computed in parallel (by increasing `MaxProcesses` parameters in script) but this is not recommended as computations can be memory intensive. To run timings for WRMG run

```bash
python3 {problem}_generator.py
```

and for the direct solver run

```bash
python3 {problem}_generator.py --flags 'lu'
```

Timings will be output in the terminal as well as saved to `data/` subdirectory as a csv file.

#### {problem}_vis.py

A helper function to visualise computational timings both in the terminal and to save them as csv files.

#### lid_stepper.py

A variation of `lid.py` where the space-time finite element method is solved over a single time slab sequentually in a time-stepping manor. Can be called in `lid_caller.py` with the flag `--stepper`. 

#### `lid_parallel_generator.py`

Generates parallelisation study for lid-driven cavity presented in the paper by varying the number of MPI cores for the WRMG solver. To generate the timings for the time-stepping approach pass the flag `--flags 'stepper'`.
