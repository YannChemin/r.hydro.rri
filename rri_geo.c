/* Geodesic distance (Hubeny's formula), used to compute dx/dy for a
 * lat/lon (utm==0) domain. Extracted out of the vendored engine's
 * rri_io.c (RRI_Sub.f90: hubeny_sub) so the native module links against
 * zero ASCII-I/O code, not even indirectly via an otherwise-unused
 * object file -- everything else in rri_io.c is RRI_Input.txt/ESRI-ASCII
 * parsing this module deliberately does not use. Kept byte-identical to
 * the vendored copy's implementation; if that one is ever bugfixed
 * upstream, re-copy this function too. */

#include <math.h>

double rri_hubeny_sub(double x1_deg, double y1_deg, double x2_deg, double y2_deg)
{
    const double pi = 3.1415926535897;
    double x1 = x1_deg * pi / 180.0;
    double y1 = y1_deg * pi / 180.0;
    double x2 = x2_deg * pi / 180.0;
    double y2 = y2_deg * pi / 180.0;

    double dy = y1 - y2;
    double dx = x1 - x2;
    double mu = (y1 + y2) / 2.0;

    const double a = 6378137.0000;
    const double b = 6356752.3140;

    double e = sqrt((a * a - b * b) / (a * a));
    double W = sqrt(1.0 - e * e * sin(mu) * sin(mu));
    double N = a / W;
    double M = a * (1.0 - e * e) / (W * W * W);

    return sqrt((dy * M) * (dy * M) + (dx * N * cos(mu)) * (dx * N * cos(mu)));
}
