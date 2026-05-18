/*
 * Generator for phase-2 reference tables: full cdfnor end-to-end.
 *
 * Writes (mean, sd, x, p, q) rows from cdfnor with which=1 over a small
 * grid spanning the body and tails. Verifies that the Rust Normal
 * composition matches the C cdfnor (which is, in turn, the composition
 * of cumnor with the standardization).
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

int main(void) {
    /* --------------------------------------------------------- normal cdf */
    FILE* f = open_csv("tests/data/normal_cdf.csv", "mean, sd, x, cdf, sf");
    double means[] = {0.0, -1.5, 3.25};
    double sds[]   = {1.0,  0.5, 2.75};
    for (size_t i = 0; i < sizeof(means)/sizeof(means[0]); i++) {
        double mean = means[i];
        double sd   = sds[i];
        for (double x = mean - 12.0 * sd; x <= mean + 12.0 * sd + 1e-12; x += sd * 0.125) {
            int which = 1;
            int status = 0;
            double p = 0.0, q = 0.0, xx = x, mm = mean, ss = sd, bound = 0.0;
            cdfnor(&which, &p, &q, &xx, &mm, &ss, &status, &bound);
            if (status != 0) continue;
            fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g\n", mean, sd, x, p, q);
        }
    }
    fclose(f);

    /* ------------------------------------------------------- normal inv cdf */
    /* Parametrize by x then store (mean, sd, p, q, x). At x = mean + 8·sd,
       q is ~6e-16, unrepresentable as 1-p. */
    f = open_csv("tests/data/normal_inverse_cdf.csv", "mean, sd, p, q, x");
    for (size_t i = 0; i < sizeof(means)/sizeof(means[0]); i++) {
        double mean = means[i];
        double sd   = sds[i];
        for (double x = mean - 7.0 * sd; x <= mean + 7.0 * sd + 1e-12; x += sd * 0.125) {
            int which = 1;
            int status = 0;
            double p = 0.0, q = 0.0, xx = x, mm = mean, ss = sd, bound = 0.0;
            cdfnor(&which, &p, &q, &xx, &mm, &ss, &status, &bound);
            if (status != 0) continue;
            if (p <= 0.0 || p >= 1.0 || q <= 0.0 || q >= 1.0) continue;
            fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g\n", mean, sd, p, q, x);
        }
    }
    fclose(f);

    fprintf(stderr, "wrote 2 tables under tests/data/\n");
    return 0;
}
