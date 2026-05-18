/*
 * Reference tables for the erf and standard-normal kernels:
 * error_f, error_fc, error_fc_scaled, cumnor, dinvnr.
 *
 * Build and run via tests/regenerate/regenerate.sh.
 */

#include "refs/cdflib.h"
#include <stdio.h>
#include <stdlib.h>

static FILE* open_csv(const char* relpath, const char* header) {
    FILE* f = fopen(relpath, "w");
    if (!f) {
        fprintf(stderr, "could not open %s for writing\n", relpath);
        exit(1);
    }
    fprintf(f, "# %s\n", header);
    return f;
}

static void write_row3(FILE* f, double a, double b, double c) {
    /* %.17g is the round-trip-safe printf for f64 — every distinct f64
       value gets a distinct decimal representation. */
    fprintf(f, "%.17g,%.17g,%.17g\n", a, b, c);
}

static void write_row2(FILE* f, double a, double b) {
    fprintf(f, "%.17g,%.17g\n", a, b);
}

int main(void) {
    /* ------------------------------------------------------------ error_f */
    FILE* erff = open_csv("tests/data/error_f.csv", "x, erf(x)");
    /* Dense grid spanning all three branches (|x| ≤ 0.5, ≤ 4, < 5.8, ≥ 5.8)
       plus the saturation cutoff. Step 0.0625 to land on f64-exact values. */
    for (double x = -6.0; x <= 6.0 + 1e-12; x += 0.0625) {
        double xx = x;
        write_row2(erff, x, error_f(&xx));
    }
    fclose(erff);

    /* ----------------------------------------------------------- error_fc */
    FILE* erfcf = open_csv("tests/data/error_fc.csv", "x, erfc(x)");
    int ind0 = 0;
    /* erfc needs more extreme positive coverage because the right tail is
       what we actually care about for accuracy. */
    for (double x = -6.0; x <= 30.0 + 1e-12; x += 0.0625) {
        double xx = x;
        write_row2(erfcf, x, error_fc(&ind0, &xx));
    }
    fclose(erfcf);

    /* ----------------------------------------------------- error_fc_scaled */
    FILE* erfcs = open_csv("tests/data/error_fc_scaled.csv", "x, erfc(x)*exp(x^2)");
    int ind1 = 1;
    /* Scaled form is well-defined for arbitrarily large positive x; large
       negative x explodes via 2*exp(x^2), so cap the negative grid. */
    for (double x = -6.0; x <= 60.0 + 1e-12; x += 0.0625) {
        double xx = x;
        write_row2(erfcs, x, error_fc(&ind1, &xx));
    }
    fclose(erfcs);

    /* ------------------------------------------------------------- cumnor */
    FILE* cnf = open_csv("tests/data/cumnor.csv", "x, cum, ccum");
    /* Cover all three regions of cumnor (|x| ≤ 0.66291, ≤ √32, > √32) plus
       deep tails up to |x| = 38 where the result is near machine min. */
    for (double x = -38.0; x <= 38.0 + 1e-12; x += 0.0625) {
        double cum, ccum;
        double xx = x;
        cumnor(&xx, &cum, &ccum);
        write_row3(cnf, x, cum, ccum);
    }
    fclose(cnf);

    /* ------------------------------------------------------------- dinvnr */
    /* Parametrize by x rather than p, then store (p, q, x). This keeps the
       small tail represented honestly: at x = 8 we have q ≈ 6e-16, which
       is unrepresentable as `1 - p` because p is rounded to 1.0. */
    FILE* dnf = open_csv("tests/data/dinvnr.csv", "p, q, x");
    for (double x = -7.0; x <= 7.0 + 1e-12; x += 0.0625) {
        double cum, ccum;
        double xx = x;
        cumnor(&xx, &cum, &ccum);
        /* Skip rows where cum or ccum has saturated to exactly 0 or 1,
           since they have no information for the inverse. */
        if (cum <= 0.0 || cum >= 1.0 || ccum <= 0.0 || ccum >= 1.0) continue;
        double back = dinvnr(&cum, &ccum);
        write_row3(dnf, cum, ccum, back);
    }
    fclose(dnf);

    fprintf(stderr, "wrote 5 tables under tests/data/\n");
    return 0;
}
