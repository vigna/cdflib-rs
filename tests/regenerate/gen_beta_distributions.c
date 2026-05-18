/*
 * Generator for phase-8 reference tables: Beta, Student's t, F distributions.
 *
 * Each table records the CDF values of the distribution at a parameter
 * grid that exercises the underlying beta_inc regimes (small/large a,
 * small/large b, tail vs body).
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
    /* --------------------------------------------------------- Beta CDF
       cumbet writes (cum, ccum) for the Beta(a,b) CDF at x in (0,1). */
    FILE* f = open_csv("tests/data/beta_cdf.csv",
                       "a, b, x, cdf, sf");
    double ab[] = {0.5, 1.0, 2.0, 5.0, 30.0};
    for (size_t i = 0; i < sizeof(ab)/sizeof(ab[0]); i++) {
        for (size_t j = 0; j < sizeof(ab)/sizeof(ab[0]); j++) {
            double a = ab[i], b = ab[j];
            for (double x = 0.05; x < 1.0; x += 0.05) {
                double y = 1.0 - x;
                double cum = 0.0, ccum = 0.0;
                cumbet(&x, &y, &a, &b, &cum, &ccum);
                fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g\n",
                        a, b, x, cum, ccum);
            }
        }
    }
    fclose(f);

    /* ---------------------------------------------------- Student's t CDF
       cumt at a range of df and t values. */
    f = open_csv("tests/data/students_t_cdf.csv", "df, t, cdf, sf");
    double dfs[] = {1.0, 2.0, 5.0, 10.0, 30.0, 100.0};
    for (size_t i = 0; i < sizeof(dfs)/sizeof(dfs[0]); i++) {
        double df = dfs[i];
        for (double t = -8.0; t <= 8.0 + 1e-12; t += 0.25) {
            double cum = 0.0, ccum = 0.0;
            cumt(&t, &df, &cum, &ccum);
            fprintf(f, "%.17g,%.17g,%.17g,%.17g\n", df, t, cum, ccum);
        }
    }
    fclose(f);

    /* --------------------------------------------------------- F CDF
       cumf at (dfn, dfd) × f grid. */
    f = open_csv("tests/data/f_cdf.csv", "dfn, dfd, f, cdf, sf");
    double dfs_for_f[] = {1.0, 2.0, 5.0, 10.0, 30.0};
    for (size_t i = 0; i < sizeof(dfs_for_f)/sizeof(dfs_for_f[0]); i++) {
        for (size_t j = 0; j < sizeof(dfs_for_f)/sizeof(dfs_for_f[0]); j++) {
            double dfn = dfs_for_f[i], dfd = dfs_for_f[j];
            for (double fx = 0.05; fx <= 10.0 + 1e-12; fx += 0.25) {
                double cum = 0.0, ccum = 0.0;
                cumf(&fx, &dfn, &dfd, &cum, &ccum);
                fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g\n",
                        dfn, dfd, fx, cum, ccum);
            }
        }
    }
    fclose(f);

    fprintf(stderr, "wrote 3 tables under tests/data/\n");
    return 0;
}
