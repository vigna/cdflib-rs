/*
 * Generator for phase-7 reference tables: beta function family.
 *
 * Writes (a, b, ln_beta) and (a, b, x, P, Q) tables. The (a, b, x) grid
 * is chosen to exercise every regime of beta_inc.
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
    /* ------------------------------------------------------ beta_log */
    FILE* f = open_csv("tests/data/beta_log.csv", "a, b, ln_beta(a, b)");
    double ab[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0};
    for (size_t i = 0; i < sizeof(ab) / sizeof(ab[0]); i++) {
        for (size_t j = 0; j < sizeof(ab) / sizeof(ab[0]); j++) {
            double a = ab[i], b = ab[j];
            fprintf(f, "%.17g,%.17g,%.17g\n", a, b, beta_log(&a, &b));
        }
    }
    fclose(f);

    /* ------------------------------------------------------ beta_inc */
    f = open_csv("tests/data/beta_inc.csv", "a, b, x, P, Q");
    double abs_for_inc[] = {0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 100.0};
    for (size_t i = 0; i < sizeof(abs_for_inc) / sizeof(abs_for_inc[0]); i++) {
        for (size_t j = 0; j < sizeof(abs_for_inc) / sizeof(abs_for_inc[0]); j++) {
            double a = abs_for_inc[i], b = abs_for_inc[j];
            for (double x = 0.05; x < 1.0; x += 0.05) {
                double y = 1.0 - x;
                double w = 0.0, w1 = 0.0;
                int ierr = 0;
                beta_inc(&a, &b, &x, &y, &w, &w1, &ierr);
                if (ierr != 0) continue;
                fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g\n", a, b, x, w, w1);
            }
        }
    }
    fclose(f);

    fprintf(stderr, "wrote 2 tables under tests/data/\n");
    return 0;
}
