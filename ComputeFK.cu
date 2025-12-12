#include "../PreCompiledHeader/FKpch.h"
#include "../Structures/StructureManagment.h"
#include "./ComputeFK.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Allocate space to grid
GridData::GridData(int w, int h) : width(w), height(h) {
    size_t size = w * h * sizeof(double);
    cudaMalloc(&vlt_vals, size);
    cudaMalloc(&fig_vals, size);
    cudaMalloc(&sig_vals, size);
}

// Deconstruct grid and free memory
GridData::~GridData() {
    cudaFree(vlt_vals);
    cudaFree(fig_vals);
    cudaFree(sig_vals);
}

// Construct grid data
GridData::GridData(GridData&& other) noexcept
    : width(other.width), height(other.height),
    vlt_vals(other.vlt_vals), fig_vals(other.fig_vals), sig_vals(other.sig_vals) {
    other.vlt_vals = nullptr;
    other.fig_vals = nullptr;
    other.sig_vals = nullptr;
}

// Allow the = operator to be used with GridData
GridData& GridData::operator=(GridData&& other) noexcept {
    if (this != &other) {
        cudaFree(vlt_vals);
        cudaFree(fig_vals);
        cudaFree(sig_vals);
        width = other.width;
        height = other.height;
        vlt_vals = other.vlt_vals;
        fig_vals = other.fig_vals;
        sig_vals = other.sig_vals;
        other.vlt_vals = nullptr;
        other.fig_vals = nullptr;
        other.sig_vals = nullptr;
    }
    return *this;
}

// Given i and j coordinates, fid the GridData equivalent index
int GridData::index(int i, int j) const {
    return j * width + i;
}

// Copy grid data from the device using CUDA
void GridData::copyFromHost(double* h_vlt, double* h_fig, double* h_sig) const {
    cudaMemcpy(vlt_vals, h_vlt, width * height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(fig_vals, h_fig, width * height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sig_vals, h_sig, width * height * sizeof(double), cudaMemcpyHostToDevice);
}

// Copy grid data to the device using CUDA
void GridData::copyToHost(double* h_vlt, double* h_fig, double* h_sig) const {
    cudaMemcpy(h_vlt, vlt_vals, width * height * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fig, fig_vals, width * height * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sig, sig_vals, width * height * sizeof(double), cudaMemcpyDeviceToHost);
}

// Initialize the data within GridData
GridData StartSim(GridData& grid, const double vlt_start, const double fig_start, const double sig_start) {
    int width = grid.width;
    int height = grid.height;
    int totalSize = width * height;

    // Allocate host memory
    std::vector<double> h_vlt(totalSize, vlt_start);
    std::vector<double> h_fig(totalSize, fig_start);
    std::vector<double> h_sig(totalSize, sig_start);

    // Copy to device
    grid.copyFromHost(h_vlt.data(), h_fig.data(), h_sig.data());

    return std::move(grid);
}

// Define a heaviside function
inline double heaviside_ge(double x, double thr) { return (x >= thr) ? 1.0 : 0.0; }

// Define a delta function
inline double indicator_eq(double x, double thr) { return (x == thr) ? 1.0 : 0.0; }

// Create function with base Fenton-Karma equations to use later
std::array<double, 3> base_equations(double vlt, double fig, double sig, const Parameters& params) {
    const double p = heaviside_ge(vlt, params.V_c);
    const double q = heaviside_ge(vlt, params.V_v);
    const double tau_mv = (1.0 - q) * params.tau_v1 + q * params.tau_v2;

    // Currents
    const double Ifi = -fig * p * (vlt - params.V_c) * (1.0 - vlt) / params.tau_d;
    const double Iso = (vlt * (1.0 - p) / params.tau_0) + (p / params.tau_r);
    const double Isi = -sig * (1.0 + std::tanh(params.K * (vlt - params.V_sic))) / (2.0 * params.tau_si);

    const double I_sum = Ifi + Iso + Isi;


    // Gate variable updates
    const double fig_new = fig + params.dt * ((1.0 - p) * (1.0 - fig) / tau_mv - p * fig / params.tau_pv);
    const double sig_new = sig + params.dt * ((1.0 - p) * (1.0 - sig) / params.tau_mw - p * sig / params.tau_pw);

    // Voltage update
    const double dVlt2dt = -I_sum;
    const double vlt_new = vlt + dVlt2dt * params.dt;

    return{ vlt_new, fig_new, sig_new };
}

// Create sech function for math
double sech(double x) {
    return 1.0 / std::cosh(x);
}

// Function that intakes the model state and returns the jacobian of the state
std::array<std::array<double, 3>, 3> jac_equations(double vlt, double fig, double sig, const Parameters& params) {
    const double p = heaviside_ge(vlt, params.V_c);
    const double p_dir = indicator_eq(vlt, params.V_c);
    const double q = heaviside_ge(vlt, params.V_v);
    const double q_dir = indicator_eq(vlt, params.V_v);
    const double tau_mv = (1.0 - q) * params.tau_v1 + q * params.tau_v2;

    // Precompute sech^2 term used in derivative of tanh
    const double arg = params.K * (vlt - params.V_sic);
    const double sech2 = sech(arg) * sech(arg);


    const double j11 = (1.0) * (
        sig * (-params.K * sech2) / (2.0 * params.tau_si)
        + (fig * p * (1.0 - 2.0 * vlt - params.V_c) / params.tau_d
            + fig * p_dir * (1.0 - vlt) * (vlt - params.V_c) / params.tau_d)
        + (-(1.0 - p) / params.tau_0 - vlt * (-p_dir) / params.tau_0 + p_dir / params.tau_r)
        );

    const double j12 = p * (vlt - params.V_c) * (1.0 - vlt) / params.tau_d;
    const double j13 = (1.0 + std::tanh(arg)) / (2.0 * params.tau_si);


    const double j21 = -p_dir * (1.0 - fig) * tau_mv
        - (1.0 - p) * (1.0 - fig) * ((1.0 - q_dir) * params.tau_v1 + q_dir * params.tau_v2)
        - p_dir * fig / params.tau_pv;
    const double j22 = ((1.0 - p) * (-1.0) / tau_mv - p / params.tau_pv);
    const double j23 = 0.0;


    const double j31 = -p_dir * (1.0 - sig) / params.tau_mw - p_dir * sig / params.tau_pw;
    const double j32 = 0.0;
    const double j33 = (1.0 - p) * (-1.0) / params.tau_mw - p / params.tau_pw;

    return { {
        { j11, j12, j13 },
        { j21, j22, j23 },
        { j31, j32, j33 }
    } };
}

// Find the residuals of the model state
std::array<double, 3> residuals(double vlt, double fig, double sig, const Parameters& params) {
    const auto next = base_equations(vlt, fig, sig, params);
    return { next[0] - vlt, next[1] - fig, next[2] - sig };
}

// Small 3x3 linear solver (Gaussian elimination), needed for math
bool solve3x3(const std::array<std::array<double, 3>, 3>& A, const std::array<double, 3>& b, std::array<double, 3>& x) {
    // Augmented matrix
    double M[3][4] = {
        { A[0][0], A[0][1], A[0][2], b[0] },
        { A[1][0], A[1][1], A[1][2], b[1] },
        { A[2][0], A[2][1], A[2][2], b[2] }
    };

    // Forward elimination with partial pivoting
    for (int i = 0; i < 3; ++i) {

        // Pivot
        int piv = i;
        for (int r = i + 1; r < 3; ++r) {
            if (std::fabs(M[r][i]) > std::fabs(M[piv][i])) piv = r;
        }
        if (std::fabs(M[piv][i]) < 1e-16) return false; // singular or ill-conditioned

        // Swap rows
        if (piv != i) {
            for (int c = i; c < 4; ++c) std::swap(M[i][c], M[piv][c]);
        }

        // Normalize and eliminate
        double diag = M[i][i];
        for (int c = i; c < 4; ++c) M[i][c] /= diag;
        for (int r = i + 1; r < 3; ++r) {
            double f = M[r][i];
            for (int c = i; c < 4; ++c) M[r][c] -= f * M[i][c];
        }
    }

    // Back substitution
    for (int i = 2; i >= 0; --i) {
        double sum = M[i][3];
        for (int c = i + 1; c <= 2; ++c) sum -= M[i][c] * x[c];
        x[i] = sum;
    }

    return true;
}

// Find magnitude to normalize to
double norm2 (std::array<double, 3> v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
};

// Newtonian solver
std::array<double, 3> locate_tip_phase(const Parameters& params, std::array<double, 3> init, int max_iters, double tol) {
    std::array<double, 3> x = init;

    for (int iter = 0; iter < max_iters; ++iter) {
        auto r = residuals(x[0], x[1], x[2], params);
        double rnorm = norm2(r);
        if (rnorm < tol) break;

        auto J = jac_equations(x[0], x[1], x[2], params);


        // Solve J * dx = -r
        std::array<double, 3> dx, rhs = { -r[0], -r[1], -r[2] };
        if (!solve3x3(J, rhs, dx)) {
            // If Jacobian is singular, exit or try a tiny step
            dx = { 0.0, 0.0, 0.0 };
            break;
        }

        // Backtracking line search
        double alpha = 1.0;
        const double c = 1e-4;
        const int max_bt = 12;
        std::array<double, 3> x_new;
        for (int bt = 0; bt < max_bt; ++bt) {
            x_new = { x[0] + alpha * dx[0],
                      x[1] + alpha * dx[1],
                      x[2] + alpha * dx[2] };

            auto r_new = residuals(x_new[0], x_new[1], x_new[2], params);
            double rnorm_new = norm2(r_new);


            if (rnorm_new <= (1.0 - c * alpha) * rnorm) {
                break; // sufficient decrease
            }
            alpha *= 0.5;
        }

        x = x_new;

        if (norm2(dx) * alpha < tol) break;
    }

    return x;
}


// CUDA Kernel
//  Apply the first stimulus to the system (a plane wave propogating in the X direction from left to right)
__global__ void applyS1Perp(double* vlt_in, int width, int height, Parameters params, double time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;


    int col = idx % width;

    if (col <= static_cast<int>(width/10)) {
        vlt_in[idx] = 1.0f;
    }

}

// Keep angle between 0 and 2pi, and alligned with polar axis lines
double AngleWrap(double angle) {

    double diff = std::fmod(angle + M_PI, 2 * M_PI);
    if (diff < 0) {
        diff = diff + 2.0 * M_PI;
    }
    return diff - M_PI;

}

// Using the phase of the spiral wave, detect the tip of the spiral wave (phase method)
std::pair<int, int> PhaseTipDetection(std::vector<double>phase_vals, int width, int height) {
    int total = width * height;
    std::vector<double> wind_vals(total, 1.0);

    // To detect the phase around a singularity (as techinically its phase is undefined) we must calculate the "wind" of the points around our target point
    for (int idx = 0; idx < total; ++idx) {
        int i = idx % width;
        int j = idx / width;

        int right_i = (i < width - 1) ? i + 1 : i;
        int up_j = (j > 0) ? j - 1 : j;


        int right_up_idx = up_j * width + right_i;
        int right_idx = j * width + right_i;
        int up_idx = up_j * width + i;

        double phase_right = AngleWrap(phase_vals[right_idx] - phase_vals[idx]);
        double phase_up = AngleWrap(phase_vals[right_up_idx] - phase_vals[right_idx]);
        double phase_left = AngleWrap(phase_vals[up_idx] - phase_vals[right_up_idx]);
        double phase_down = AngleWrap(phase_vals[idx] - phase_vals[up_idx]);

        wind_vals[idx] = phase_right + phase_up + phase_left + phase_down;
    }

    // Find the wind value that is closest to 2pi, this will be our singularity point
    double targ_wind = 0.0;
    double targ_min = 2.0 * M_PI;
    int idx_targ_last = 0;
    for (int l = 0; l < total; ++l) {
        double tester = abs(abs(wind_vals[l]) - 2.0 * M_PI);
        if (tester < targ_min) {
            targ_wind = abs(wind_vals[l]);
            targ_min = tester;
            idx_targ_last = l;
        }
    }

    // Assign the integer values of the spiral wave tip location coordinates
    int i_targ = idx_targ_last % width;
    int j_targ = idx_targ_last / width;

    // Safe guard
    if (targ_wind > M_PI) {
        return std::make_pair(i_targ, j_targ);
    }

    // Catch NAN values
    return std::make_pair(-1, -1);
}

// Using the voltage and gating variable of the spiral wave, detect the tip of the spiral wave (contour method)
std::pair<int, int> ApproxTip(std::vector<double>val_vec1, std::vector<double>val_vec2, double val_tar1, double val_tar2, int width, int height) {
    int total = width * height;

    // Find the squared distance of the grid values from the targeted u and v values for the spiral wave tip
    std::vector<double> dist_abs(total);
    for (int ii = 0; ii < total; ++ii) {
        dist_abs[ii] = (abs(val_vec1[ii] - val_tar1)) + (abs(val_vec2[ii] - val_tar2));
    }

    // Find the minimum squared distance from the target values
    double dist_abs_min_last = 100.0;
    int idx_min_last = 10;
    for (int idx = 0; idx < total; ++idx) {
        if (dist_abs[idx] < dist_abs_min_last) {
            dist_abs_min_last = dist_abs[idx];
            idx_min_last = idx;
        }
    }

    // Assign the integer values of the spiral wave tip location coordinates
    int i = idx_min_last % width;
    int j = idx_min_last / width;

    // Safeguard, catch anything outside of reason
    if (i > 0 && i < width && j > 0 && j < height && dist_abs_min_last > 0.0 && dist_abs_min_last < 0.01) {
        return std::make_pair(i, j);
    }

    // Catch NAN values
    return std::make_pair(-1, -1);
}

// Using the voltage of the spiral wave at two different times, detect the tip of the spiral wave (Jacobian Determinate Method)
std::pair<int, int> Jacobian_Determinate_Method(std::vector<double>val_vec_t1, std::vector<double>val_vec_t2, int width, int height) {
    int total = width * height;

    // Take the determinate of the Jacobian around each point
    std::vector<double> DVx(total);
    for (int idx = 0; idx < total; ++idx) {

        int i = idx % width;
        int j = idx / width;

        int left_i = (i > 0) ? i - 1 : i;
        int right_i = (i < width - 1) ? i + 1 : i;
        int up_j = (j > 0) ? j - 1 : j;
        int down_j = (j < height - 1) ? j + 1 : j;

        int left_idx = j * width + left_i;
        int right_idx = j * width + right_i;
        int up_idx = up_j * width + i;
        int down_idx = down_j * width + i;

        double DVx_temp = (val_vec_t1[right_idx] - val_vec_t1[left_idx]) / (right_i - left_i) * (val_vec_t2[up_idx] - val_vec_t2[down_idx]) / (up_j - down_j) - (val_vec_t1[up_idx] - val_vec_t1[down_idx]) / (up_j - down_j) * (val_vec_t2[right_idx] - val_vec_t2[left_idx]) / (right_i - left_i);
        DVx[idx] = DVx_temp;

    }


    // Find the greatest value of DVx
    double DVx_max_last = 0.0;
    int idx_max_last = 10;

    for (int idx = 0; idx < total; ++idx) {
        if (DVx[idx] > DVx_max_last) {
            DVx_max_last = DVx[idx];
            idx_max_last = idx;
        }
    }

    // Assign the integer values of the spiral wave tip location coordinates
    int i = idx_max_last % width;
    int j = idx_max_last / width;

    // Safeguard, check that the spiral wave tip is not at (or close to) the boundaries
    if (i > 5 && i < width - 5 && j > 5 && j < height - 5) {
        return std::make_pair(i, j);
    }

    // Catch NAN values
    return std::make_pair(-1, -1);
}

// CUDA Kernel
//  Apply the second stimulus to the system (a plane wave propogating in the Y direction from bottom to top)
__global__ void applyS2Perp(double* vlt_in, int width, int height, Parameters params, double time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;


    int row = idx / width;

    if (row >= static_cast<int>(height - 0.5 * height)) {
        vlt_in[idx] = 1.0f;
    }

}

// CUDA Kernel
// The standard equations used to calculate the next time step of the Fienton-Karma equations based on the previous time step
__global__ void fentonKarmaKernel(double* vlt_in, double* fig_in, double* sig_in, double* vlt_out, double* fig_out, double* sig_out, int width, int height, Parameters params, double time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx >= total) return;

    int i = idx % width;
    int j = idx / width;

    // Compute safe neighbor indices
    int left_i = (i > 0) ? i - 1 : i;
    int right_i = (i < width - 1) ? i + 1 : i;
    int up_j = (j > 0) ? j - 1 : j;
    int down_j = (j < height - 1) ? j + 1 : j;

    // Convert to array indexing
    int left_idx = j * width + left_i;
    int right_idx = j * width + right_i;
    int up_idx = up_j * width + i;
    int down_idx = down_j * width + i;


    // Define centers of cells
    double vlt_center = vlt_in[idx];
    double fig_center = fig_in[idx];
    double sig_center = sig_in[idx];

    // Preform laplacian
    double vlt_left = vlt_in[left_idx];
    double vlt_right = vlt_in[right_idx];
    double vlt_up = vlt_in[up_idx];
    double vlt_down = vlt_in[down_idx];

    double lap_vlt = (vlt_left + vlt_right + vlt_up + vlt_down - 4.0f * vlt_center) / (params.dx * params.dx);

    // Fenton Karma equations
    double p = (vlt_center >= params.V_c);
    double q = (vlt_center >= params.V_v);
    double tau_mv = (1.0 - q) * params.tau_v2 + q * params.tau_v1;

    double Ifi = (- 1.0 * fig_center * p * (vlt_center - params.V_c) * (1.0 - vlt_center)) / params.tau_d;
    double Iso = (vlt_center * (1.0 - p) / params.tau_0) + (p / params.tau_r);
    double Isi = (- 1.0 * sig_center * (1.0 + tanh(params.K * (vlt_center - params.V_sic)))) / (2.0 * params.tau_si);

    double I_sum = Ifi + Iso + Isi;

    double dVltdt = lap_vlt * params.Diff_Coef;
    dVltdt = dVltdt - I_sum;
    double dFigdt = ((1.0 - p) * (1.0 - fig_center) / tau_mv) - (p * fig_center / params.tau_pv);
    double dSigdt = ((1.0 - p) * (1.0 - sig_center) / params.tau_mw) - (p * sig_center / params.tau_pw);

    // Update the output
    vlt_out[idx] = vlt_center + params.dt * dVltdt;
    fig_out[idx] = fig_center + params.dt * dFigdt;
    sig_out[idx] = sig_center + params.dt * dSigdt;

}

/**
 * Wrapper function for the CUDA kernel function.
 */
// Progress the Fenton-Karma model according to simulation specifications
void evolveFentonKarma(GridData& grid, const Parameters& params, SimConstraints& simulation, int steps) {
    int total = grid.width * grid.height;
    size_t size = total * sizeof(double);

    // Allocate memory to the current and next step of the function
    double* vlt1, * fig1, * sig1, * vlt2, * fig2, * sig2;
    cudaMalloc(&vlt1, size);
    cudaMalloc(&fig1, size);
    cudaMalloc(&sig1, size);
    cudaMalloc(&vlt2, size);
    cudaMalloc(&fig2, size);
    cudaMalloc(&sig2, size);

    // Assign values to the current step of the function
    cudaMemcpy(vlt1, grid.vlt_vals, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(fig1, grid.fig_vals, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(sig1, grid.sig_vals, size, cudaMemcpyDeviceToDevice);

    // Set up for CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    // Set up for tip tracking
    int tip_track_start_step = static_cast<int>(params.spiral_time / params.dt) + static_cast<int>(10 / params.dt);
    int tip_track_steps = (steps + 1) - tip_track_start_step;
    size_t tip_track_size = tip_track_steps * sizeof(double);
    std::vector<int> tip_traj_JDM_x(tip_track_steps);
    std::vector<int> tip_traj_JDM_y(tip_track_steps);
    std::vector<double> tip_traj_volt_x(tip_track_steps);
    std::vector<double> tip_traj_volt_y(tip_track_steps);
    std::vector<double> tip_traj_phase_x(tip_track_steps);
    std::vector<double> tip_traj_phase_y(tip_track_steps);

    // Set up for phase conversion
    std::array<double, 3> singularity_phase = locate_tip_phase(params);
    double vlt_singularity = singularity_phase[0];
    double fig_singularity = singularity_phase[1];
    double sig_singularity = singularity_phase[2];
    printf("vlt_sing = %f, fig_sing = %f, sig_sing = %f\n", vlt_singularity, fig_singularity, sig_singularity);

    // For each time step evolve the simulation using the Fenton-Karma Model
    for (int step = 0; step <= steps; ++step) {
        double t = step * params.dt;

        // Apply S1 stimulus to the simulation
        if (simulation.spiral && step == 0) {
            applyS1Perp CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (
                vlt1, grid.width, grid.height, params, t
                );
        }

        // Apply S2 stimulus to the simulation
        if (simulation.spiral && step == params.spiral_time / params.dt) {
            applyS2Perp CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (
                vlt1, grid.width, grid.height, params, t
                );
        }

        // Calculate next time step
        fentonKarmaKernel CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (
            vlt1, fig1, sig1, vlt2, fig2, sig2, grid.width, grid.height, params, t
            );

        // Safeguard
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }

        // Syncronize calculations using CUDA
        cudaDeviceSynchronize();

        // Update time
        double t_new = t + params.dt;

        // Apply and append to arrays for JDM tip tracking
        if ((simulation.tip_track_JDM) && step >= tip_track_start_step) {
            std::vector<double> host_vlt_old(total + 1);
            std::vector<double> host_vlt_new(total + 1);

            cudaMemcpy(host_vlt_old.data(), vlt1, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_vlt_new.data(), vlt2, size, cudaMemcpyDeviceToHost);

            std::pair<int, int> ij_est = Jacobian_Determinate_Method(host_vlt_old, host_vlt_new, grid.width, grid.height);
            int i_est = std::get<0>(ij_est);
            int j_est = std::get<1>(ij_est);

            tip_traj_JDM_x[static_cast<int>(step - tip_track_start_step)] = i_est;
            tip_traj_JDM_y[static_cast<int>(step - tip_track_start_step)] = j_est;
        }

        // Apply and append to arrays for contour tip tracking
        if ((simulation.tip_track_volt) && step >= tip_track_start_step) {

            std::vector<double> host_vlt_new(total + 1);
            std::vector<double> host_fig_new(total + 1);

            cudaMemcpy(host_vlt_new.data(), vlt2, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_fig_new.data(), fig2, size, cudaMemcpyDeviceToHost);

            std::pair<int, int> ij_approx = ApproxTip(host_vlt_new, host_fig_new, params.vlt_tip_pick, params.fig_tip_pick, grid.width, grid.height);
            int i_approx = std::get<0>(ij_approx);
            int j_approx = std::get<1>(ij_approx);

            if (i_approx > 0 && j_approx > 0 && i_approx < grid.width && j_approx < grid.height) {
                tip_traj_volt_x[static_cast<int>(step - tip_track_start_step)] = i_approx;
                tip_traj_volt_y[static_cast<int>(step - tip_track_start_step)] = j_approx;
            }


        }

        // Apply and append to arrays for phase tip tracking
        if ((simulation.tip_track_phase) && step >= tip_track_start_step) {
            std::vector<double> host_vlt_new(total + 1);
            std::vector<double> host_fig_new(total + 1);

            cudaMemcpy(host_vlt_new.data(), vlt2, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_fig_new.data(), fig2, size, cudaMemcpyDeviceToHost);

            std::vector<double> phase_vals(total, 1.0);

            for (int k = 0; k < total; ++k) {
                double uu = host_vlt_new.data()[k];
                double vv = host_fig_new.data()[k];

                phase_vals[k] = atan2(vv - fig_singularity, uu - vlt_singularity);
            }

            std::pair<int, int>  ij_approx = PhaseTipDetection(phase_vals, grid.width, grid.height);
            double i_approx = std::get<0>(ij_approx);
            double j_approx = std::get<1>(ij_approx);

            if (i_approx > 0 && j_approx > 0 && i_approx < grid.width && j_approx < grid.height) {
                tip_traj_phase_x[static_cast<int>(step - tip_track_start_step)] = i_approx;
                tip_traj_phase_y[static_cast<int>(step - tip_track_start_step)] = j_approx;
            }


        }

        // Every 100ms in the simulation, create an output file for the vlt, fig, and sig values at that time
        if ((static_cast<int>(t) % 100 == 0) || (static_cast<int>(t) == simulation.last_time)) {
            std::vector<double> host_vlt(total);
            std::vector<double> host_fig(total);
            std::vector<double> host_sig(total);

            // Copy grid values to the device
            cudaMemcpy(host_vlt.data(), vlt1, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_fig.data(), fig1, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_sig.data(), sig1, size, cudaMemcpyDeviceToHost);

            // Create target directory
            std::filesystem::create_directories("Results/" + params.run_name);

            // Writing results to file
            std::string vlt_filename = "Results/" + params.run_name + "/vlt_vals_t" + std::to_string(static_cast<int>(t)) + ".bin";
            std::string fig_filename = "Results/" + params.run_name + "/fig_vals_t" + std::to_string(static_cast<int>(t)) + ".bin";
            std::string sig_filename = "Results/" + params.run_name + "/sig_vals_t" + std::to_string(static_cast<int>(t)) + ".bin";

            std::ofstream vltfile(vlt_filename, std::ios::binary);
            std::ofstream figfile(fig_filename, std::ios::binary);
            std::ofstream sigfile(sig_filename, std::ios::binary);

            vltfile.write(reinterpret_cast<const char*>(host_vlt.data()), size);
            figfile.write(reinterpret_cast<const char*>(host_fig.data()), size);
            sigfile.write(reinterpret_cast<const char*>(host_sig.data()), size);

            vltfile.close();
            figfile.close();
            sigfile.close();
        }


        // Swap in and out vectors to continue to progress stepping function
        std::swap(vlt1, vlt2);
        std::swap(fig1, fig2);
        std::swap(sig1, sig2);
    }

    // Write JDM tip tacking values to file
    if (simulation.tip_track_JDM) {

        std::string tip_x_JDM_filename = "Results/" + simulation.run_name + "/JDM_tip_x_tracker.bin";
        std::string tip_y_JDM_filename = "Results/" + simulation.run_name + "/JDM_tip_y_tracker.bin";

        std::ofstream tip_x_JDM_file(tip_x_JDM_filename, std::ios::binary);
        std::ofstream tip_y_JDM_file(tip_y_JDM_filename, std::ios::binary);

        tip_x_JDM_file.write(reinterpret_cast<const char*>(&tip_traj_JDM_x[0]), tip_track_size);
        tip_y_JDM_file.write(reinterpret_cast<const char*>(&tip_traj_JDM_y[0]), tip_track_size);


        tip_x_JDM_file.close();
        tip_y_JDM_file.close();
    }

    // Write contour tip tacking values to file
    if (simulation.tip_track_volt) {

        std::string tip_x_volt_filename = "Results/" + simulation.run_name + "/volt_tip_x_tracker.bin";
        std::string tip_y_volt_filename = "Results/" + simulation.run_name + "/volt_tip_y_tracker.bin";

        std::ofstream tip_x_volt_file(tip_x_volt_filename, std::ios::binary);
        std::ofstream tip_y_volt_file(tip_y_volt_filename, std::ios::binary);

        tip_x_volt_file.write(reinterpret_cast<const char*>(&tip_traj_volt_x[0]), tip_track_size);
        tip_y_volt_file.write(reinterpret_cast<const char*>(&tip_traj_volt_y[0]), tip_track_size);


        tip_x_volt_file.close();
        tip_y_volt_file.close();
    }

    // Write phase tip tacking values to file
    if (simulation.tip_track_phase) {

        std::string tip_x_phase_filename = "Results/" + simulation.run_name + "/phase_tip_x_tracker.bin";
        std::string tip_y_phase_filename = "Results/" + simulation.run_name + "/phase_tip_y_tracker.bin";

        std::ofstream tip_x_phase_file(tip_x_phase_filename, std::ios::binary);
        std::ofstream tip_y_phase_file(tip_y_phase_filename, std::ios::binary);

        tip_x_phase_file.write(reinterpret_cast<const char*>(&tip_traj_phase_x[0]), tip_track_size);
        tip_y_phase_file.write(reinterpret_cast<const char*>(&tip_traj_phase_y[0]), tip_track_size);


        tip_x_phase_file.close();
        tip_y_phase_file.close();
    }

    // Copy the  memory of the end step calculated values to the device
    cudaMemcpy(grid.vlt_vals, vlt1, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(grid.fig_vals, fig1, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(grid.sig_vals, sig1, size, cudaMemcpyDeviceToDevice);

    // Clean up memory
    cudaFree(vlt1); cudaFree(fig1); cudaFree(sig1);
    cudaFree(vlt2); cudaFree(fig2); cudaFree(sig2);
}


