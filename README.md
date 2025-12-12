# Fenton-Karma Spiral Wave Tracking: A Methods Comparison

Compare and Contrast the Contour and Phase Identification Methods for Spiral Wave Identification using the Fenton-Karma Model

## Code instructions

All togglable variables and adjustable parameters are located in the `main.cpp` file, you will want to adjust variables before compilation

Highlights:

- SimConstraints simulation.last_time: Tells the simulation how many milliseconds to run for
- SimConstraints simulation.spiral: If true the simulation will apply a stimulous at the time specifiec by SimConstraints simulation.spiral_time in order to create a singular spiral wave
- SimConstraints simulation.tip_track_JDM: If true, the program will track the approximate location of the spiral wave tip using the Jacobian Determinate Method
- SimConstraints simulation.tip_track_volt: If true, the program will track the approximate location of the spiral wave tip based on the estimated value of the spiral wave tip in phase space currently stored in Parameters params.utip_pick and Parameters params.vtip_pick.
- SimConstraints simulation.tip_track_phase: If true, the program will track the approximate location of the spiral wave tip on the estimated value phase angle of the wave about the phase space calculated midpoint of the system point.
- SimConstraints.run_name: Stores the title of your simulation, your output files will be stored in a file sharing this name

## Output Files

This code outputs .bin files for the u and v state values of the Fenton-Karma Model accross the simulated region every 100ms. If tip-tracking of any kind is toggled to true there is an additional output of the x and y locations of the tip within the region, NAN values in this file are reported as x=-1 and y=-1. Included in this repository are two python codes used to visualize the results. `EvolvingPlot.py` gerenates a .gif file of the simulation which shows the evoluution of the wave in the simulated region. `Snapshot.py` reads the outputs containing the (x,y) coordinates of the spiral wave tip and then plots the results on a grid. Both python files need to have the same dimension variable values as the simulation in order to function properly. Each file is designed to be run in terminal in the `ResultingVisuals` directory using the command `python <filename>.py`. They will each create their figures in the `Figures` folder in the main directory.

## Compilation instructions

**Both compilation options require CUDA and a GPU**

**Preferred Compilation**

If you have ncvv installed run the following command in the repository directory:
```
nvcc main.cpp CUDAfiles/ComputeFK.cu PreCompiledHeader/FKpch.cpp -o fkModel.exe
```
**-OR-**

**Backup Compilation**

Run the following commands in the `CMakeLocalBuild` directory:
 ```
mkdir build && cd build
cmake ..
cmake --build .
```
 
The `fkModel.exe` file will then be in a subfolder entitled `build/run/`

## Running the Code

This code is designed to be run on the Georgia Tech PACE cluster, but can be run on any computer with a NVIDIA GPU and CUDA installed. To run locally, either open the constructed `fkModel.exe` file or use your terminal to run the program by navigating to the `fkModel.exe` and run either the `start fkModel.exe` command (Windows) or the `wine fkModel.exe` command (Linux and macOS).
