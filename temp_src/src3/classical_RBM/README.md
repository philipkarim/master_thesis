# Restricted Boltzmann Machine for FYS4411
## Code structure
### Main program
All parameters are defined in `main`. The most fundamental parts of the code are `system.cpp` which are executing the Boltzmann machine cycles and `sampler.cpp` which is sampling the results. The codes in the `initialStates` folder are initializing the visible and hidden nodes and biases in addition to the weight matrix. The codes in the `wavefunction` folder are implementing the wavefunction which is given by the quantum neural state. The codes in the `Hamiltonian` folder is used to compute the hamiltonian energy.

### Analytics
The codes used for plotting the results are `plotting.py` which plots all the figures. The last file used for analytics are `statisticalhandling.py` which uses the blocking method written by Marius Jonsson, and can be found [`here`](https://github.com/computative/block/blob/master/python/tictoc.py).

## Compiling the project using CMake:
You can install CMake through one of the Linux package managers, e.g., `apt install cmake`, `pacman -S cmake`, etc. For Mac you can install using `brew install cmake`. Other ways of installing are shown here: [https://cmake.org/install/](https://cmake.org/install/).

In a Linux/Mac terminal this can be done by the following commands
```bash
# Create build-directory
mkdir build

# Move into the build-directory
cd build

# Run CMake to create a Makefile
cmake ../

# Make the Makefile using two threads
make -j2

# Move the executable to the top-directory
mv rbm ..
```
Or, simply run the script `compile_project` via
```bash
./compile_project
```
and the same set of commands are done for you. Now the project can be run by executing
```bash
./rbm
```
in the top-directory.

#### Cleaning the directory
Run `make clean` in the top-directory to remove the executable `rbm` and the `build`-directory.

#### Armadillo
To run the project, armadillo version 10.5.0 needs to be installed.
