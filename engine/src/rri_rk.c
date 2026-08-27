/**
 * @file rri_rk.c
 * @brief Sole definition site for the Cash-Karp RK45 coefficients (see
 * rri.h: rri_rk_coeffs's doc for what these mean and why they must not
 * be re-derived from a different source). No physics/time-stepping logic
 * lives here -- just the numeric constants src/main.c's adaptive-step
 * river/slope/groundwater integrators combine every RK stage.
 *
 * Fortran reference: RRI.f90 / RRI_Mod2.f90's `runge_mod` module.
 */
#include "rri/rri.h"

void rri_rk_coeffs_init(rri_rk_coeffs *rk)
{
    rk->eps = 0.010;
    rk->ddt_min_riv = 0.10;
    rk->ddt_min_slo = 1.0;
    rk->safety = 0.90;
    rk->pgrow = -0.20;
    rk->pshrnk = -0.250;
    rk->errcon = 1.89e-4;

    rk->b21 = 0.20;
    rk->b31 = 3.0 / 40.0;  rk->b32 = 9.0 / 40.0;
    rk->b41 = 0.30;         rk->b42 = -0.90;        rk->b43 = 1.20;
    rk->b51 = -11.0 / 54.0; rk->b52 = 2.50;         rk->b53 = -70.0 / 27.0; rk->b54 = 35.0 / 27.0;
    rk->b61 = 1631.0 / 55296.0; rk->b62 = 175.0 / 512.0; rk->b63 = 575.0 / 13824.0;
    rk->b64 = 44275.0 / 110592.0; rk->b65 = 253.0 / 4096.0;
    rk->c1 = 37.0 / 378.0; rk->c3 = 250.0 / 621.0; rk->c4 = 125.0 / 594.0; rk->c6 = 512.0 / 1771.0;
    rk->dc1 = rk->c1 - 2825.0 / 27648.0;
    rk->dc3 = rk->c3 - 18575.0 / 48384.0;
    rk->dc4 = rk->c4 - 13525.0 / 55296.0;
    rk->dc5 = -277.0 / 14336.0;
    rk->dc6 = rk->c6 - 0.250;
}
