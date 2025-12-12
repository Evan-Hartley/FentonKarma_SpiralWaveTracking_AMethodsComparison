#include "PreCompiledHeader/FKpch.h"

#include "Structures/StructureManagment.h"
#include "CUDAfiles/ComputeFK.cuh"

int main() {
    // Adjustable simulation variables
    SimConstraints simulation;
    simulation.nx = 512;
    simulation.ny = 512;
    simulation.dx = 1;
    simulation.dt = 0.2;
    simulation.last_time = 1000;
    simulation.spiral = true;
    simulation.spiral_time = 400;
    simulation.tip_track_JDM = true;
    simulation.tip_track_volt = true;
    simulation.tip_track_phase = true;
    simulation.run_name = "Test";

    // Adjustable model parameters
    Parameters params;
    params.dx = 0.031;
    params.dt = 0.1;
    params.Diff_Coef = 0.001;
    params.tau_pv = 3.33;
    params.tau_v1 = 1000.0,
    params.tau_v2 = 19.2,
    params.tau_pw = 667.0,
    params.tau_mw = 11.0,
    //tau_d = C_m/g_fi
    params.tau_d = 0.25,
    params.tau_0 = 8.3,
    params.tau_r = 50.0,
    params.tau_si = 44.84,
    params.K = 10.0,
    params.V_sic = 0.85,
    params.V_c = 0.13,
    params.V_v = 0.055,
    params.last_step = static_cast<int>(simulation.last_time / params.dt);
    params.spiral_time = simulation.spiral_time;
    params.run_name = simulation.run_name;
    params.vlt_tip_pick = 0.35;              // User guess for phase singularity's vlt value
    params.fig_tip_pick = 0.35;              // User guess for phase singularity's fig value
    

    // Check for parameter cohesion
    if (params.dt > (params.dx * params.dx) / (4.0 * params.Diff_Coef)) {
        std::cerr << "Warning: dt may be too large for stability!" << std::endl;
    }

    // Create and fill a data grid of intial vlt, fig, and sig conditions
    const int width = simulation.nx, height = simulation.ny;
    GridData grid = GridData(width, height);
    grid = StartSim(grid, 0.0f, 1.0f, 1.0f);

    // Initialize vlt, fig, and sig on host
    std::vector<double> h_vlt(width * height);
    std::vector<double> h_fig(width * height);
    std::vector<double> h_sig(width * height);
    grid.copyToHost(h_vlt.data(), h_fig.data(), h_sig.data());

    // Evolve the parameters using the FK model
    evolveFentonKarma(grid, params, simulation, static_cast<int>(params.last_step));

    // Return data to host
    grid.copyToHost(h_vlt.data(), h_fig.data(), h_sig.data());

    // End
    return 0;
}