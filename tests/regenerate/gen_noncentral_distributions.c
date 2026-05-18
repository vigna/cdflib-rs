/*
 * Reference tables for the noncentral chi-squared and noncentral F
 * distributions (cumchn, cumfnc).
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

int main(void) {
    /* ------------------------------------- noncentral chi-squared CDF */
    FILE* f = open_csv("tests/data/chi_squared_noncentral_cdf.csv",
                       "df, ncp, x, cdf, sf");
    double dfs[] = {1.0, 2.0, 5.0, 10.0, 30.0};
    double ncps[] = {0.0, 0.5, 2.0, 10.0, 50.0};
    for (size_t i = 0; i < sizeof(dfs)/sizeof(dfs[0]); i++) {
        for (size_t j = 0; j < sizeof(ncps)/sizeof(ncps[0]); j++) {
            double df = dfs[i], ncp = ncps[j];
            /* Span from below the mean (df + ncp) to several SDs above. */
            double mean = df + ncp;
            double sd = (df + 2.0 * ncp <= 0.0) ? 1.0 :
                        2.0 * (df + 2.0 * ncp);
            sd = (sd <= 0.0) ? 1.0 : sd;
            sd = (sd < 1.0) ? 1.0 : sd;
            double x_max = mean + 5.0 * sd;
            double step = x_max / 30.0;
            if (step <= 0.0) step = 0.1;
            for (double x = step / 4.0; x <= x_max + 1e-12; x += step) {
                double cum = 0.0, ccum = 0.0;
                cumchn(&x, &df, &ncp, &cum, &ccum);
                fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g\n",
                        df, ncp, x, cum, ccum);
            }
        }
    }
    fclose(f);

    /* --------------------------------------------- noncentral F CDF */
    f = open_csv("tests/data/fisher_snedecor_noncentral_cdf.csv",
                 "dfn, dfd, ncp, f, cdf, sf");
    double dfns[] = {2.0, 5.0, 10.0, 30.0};
    double dfds[] = {2.0, 5.0, 10.0, 30.0};
    double ncps_f[] = {0.0, 1.0, 5.0, 20.0};
    for (size_t i = 0; i < sizeof(dfns)/sizeof(dfns[0]); i++) {
        for (size_t j = 0; j < sizeof(dfds)/sizeof(dfds[0]); j++) {
            for (size_t k = 0; k < sizeof(ncps_f)/sizeof(ncps_f[0]); k++) {
                double dfn = dfns[i], dfd = dfds[j], ncp = ncps_f[k];
                double f_max = 10.0;
                double step = 0.25;
                for (double fx = step; fx <= f_max + 1e-12; fx += step) {
                    double cum = 0.0, ccum = 0.0;
                    cumfnc(&fx, &dfn, &dfd, &ncp, &cum, &ccum);
                    fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                            dfn, dfd, ncp, fx, cum, ccum);
                }
            }
        }
    }
    fclose(f);

    fprintf(stderr, "wrote 2 tables under tests/data/\n");
    return 0;
}
