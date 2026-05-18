/*
 * Generator for phase-9 reference tables: Binomial, Poisson,
 * Negative Binomial.
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
    /* ------------------------------------------------------ Binomial CDF
       cumbin(s, n, pr, ompr) -> (cum, ccum). */
    FILE* f = open_csv("tests/data/binomial_cdf.csv", "n, pr, s, cdf, sf");
    int ns[] = {5, 10, 50, 200};
    double prs[] = {0.05, 0.25, 0.5, 0.75, 0.95};
    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); i++) {
        double n = (double) ns[i];
        for (size_t j = 0; j < sizeof(prs)/sizeof(prs[0]); j++) {
            double pr = prs[j];
            double ompr = 1.0 - pr;
            for (int s_int = 0; s_int <= ns[i]; s_int++) {
                double s = (double) s_int;
                double cum = 0.0, ccum = 0.0;
                cumbin(&s, &n, &pr, &ompr, &cum, &ccum);
                fprintf(f, "%d,%.17g,%d,%.17g,%.17g\n",
                        ns[i], pr, s_int, cum, ccum);
            }
        }
    }
    fclose(f);

    /* ------------------------------------------------------ Poisson CDF */
    f = open_csv("tests/data/poisson_cdf.csv", "lambda, s, cdf, sf");
    double lambdas[] = {0.5, 1.0, 5.0, 12.0, 50.0, 200.0};
    for (size_t i = 0; i < sizeof(lambdas)/sizeof(lambdas[0]); i++) {
        double lam = lambdas[i];
        int s_max = (int) (lam + 10.0 * (lam < 1.0 ? 1.0 : lam) + 5);
        for (int s_int = 0; s_int <= s_max; s_int++) {
            double s = (double) s_int;
            double cum = 0.0, ccum = 0.0;
            cumpoi(&s, &lam, &cum, &ccum);
            fprintf(f, "%.17g,%d,%.17g,%.17g\n", lam, s_int, cum, ccum);
        }
    }
    fclose(f);

    /* -------------------------------------------- Negative Binomial CDF
       cumnbn(s, n, pr, ompr) -> (cum, ccum). Here `s` is the number of
       failures and `n` is the target success count. */
    f = open_csv("tests/data/negative_binomial_cdf.csv",
                 "n, pr, s, cdf, sf");
    int rs[] = {1, 3, 10, 50};
    double prs_nbn[] = {0.1, 0.25, 0.5, 0.9};
    for (size_t i = 0; i < sizeof(rs)/sizeof(rs[0]); i++) {
        double r_d = (double) rs[i];
        for (size_t j = 0; j < sizeof(prs_nbn)/sizeof(prs_nbn[0]); j++) {
            double pr = prs_nbn[j];
            double ompr = 1.0 - pr;
            int s_max = (int) (rs[i] * (1.0 - pr) / pr + 10.0 *
                               (rs[i] * (1.0 - pr) / (pr * pr)) + 5);
            if (s_max < 30) s_max = 30;
            if (s_max > 1000) s_max = 1000;
            for (int s_int = 0; s_int <= s_max; s_int += 1) {
                double s = (double) s_int;
                double cum = 0.0, ccum = 0.0;
                cumnbn(&s, &r_d, &pr, &ompr, &cum, &ccum);
                fprintf(f, "%d,%.17g,%d,%.17g,%.17g\n",
                        rs[i], pr, s_int, cum, ccum);
            }
        }
    }
    fclose(f);

    fprintf(stderr, "wrote 3 tables under tests/data/\n");
    return 0;
}
