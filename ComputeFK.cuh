#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

class GridData {
public:
    int width, height;
    double* vlt_vals;
    double* fig_vals;
    double* sig_vals;

    GridData(int w, int h);
    ~GridData();

    // Delete copy constructor and copy assignment
    GridData(const GridData&) = delete;
    GridData& operator=(const GridData&) = delete;

    // Move constructor and move assignment
    GridData(GridData&& other) noexcept;
    GridData& operator=(GridData&& other) noexcept;

    int index(int i, int j) const;
    void copyFromHost(double* h_vlt, double* h_fig, double* h_sig) const;
    void copyToHost(double* h_vlt, double* h_fig, double* h_sig) const;
};

GridData StartSim(GridData& grid, const double vlt_start, const double fig_start, const double sig_start);


inline double heaviside_ge(double x, double thr);

inline double indicator_eq(double x, double thr);

std::array<double, 3> base_equations(double vlt, double fig, double sig, const Parameters& params);

double sech(double x);

std::array<std::array<double, 3>, 3> jac_equations(double vlt, double fig, double sig, const Parameters& params);

std::array<double, 3> residuals(double vlt, double fig, double sig, const Parameters& params);

bool solve3x3(const std::array<std::array<double, 3>, 3>& A, const std::array<double, 3>& b, std::array<double, 3>& x);

double norm2(std::array<double, 3> v);

std::array<double, 3> locate_tip_phase(const Parameters& params, std::array<double, 3> init = { 0.3, 0.3, 0.8 }, int max_iters = 100, double tol = 1e-4);

double AngleWrap(double angle);

std::pair<int, int> PhaseTipDetection(std::vector<double>phase_vals, int width, int height);

std::pair<int, int> ApproxTip(std::vector<double>val_vec1, std::vector<double>val_vec2, double val_tar1, double val_tar2, int width, int height);

std::pair<int, int> Jacobian_Determinate_Method(std::vector<double>val_vec_t1, std::vector<double>val_vec_t2, int width, int height);

void evolveFentonKarma(GridData& grid, const Parameters& params, SimConstraints& simulation, int steps);