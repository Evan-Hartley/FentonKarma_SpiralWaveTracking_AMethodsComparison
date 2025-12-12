#pragma once

struct Parameters {
    int last_step, spiral_time;
    double Diff_Coef, tau_pv, tau_v1, tau_v2, tau_pw, tau_mw, tau_d, tau_0, tau_r, tau_si, K, V_sic, V_c, V_v, dt, dx;
    double vlt_tip_pick, fig_tip_pick;

    std::string run_name;

    Parameters()
        : last_step(0), spiral_time(0),
        Diff_Coef(0.0f), tau_pv(0.0f), tau_v1(0.0f), tau_v2(0.0f),
        tau_pw(0.0f), tau_mw(0.0f), tau_d(0.0f), tau_0(0.0f),
        tau_r(0.0f), tau_si(0.0f), K(0.0f), V_sic(0.0f),
        V_c(0.0f), V_v(0.0f), dt(0.0f), dx(0.0f),
        vlt_tip_pick(0.0), fig_tip_pick(0.0), run_name("temp") {}

};

struct SimConstraints {
    int nx, ny, last_time, spiral_time;
    double dt, dx;
    bool spiral, tip_track_JDM, tip_track_volt, tip_track_phase;

    std::string run_name;

    SimConstraints()
        : nx(0), ny(0), dt(0.0), dx(0.0),
        last_time(0), spiral_time(0),
        spiral(false), tip_track_JDM(false), tip_track_volt(false), tip_track_phase(false), run_name("temp") {}

};