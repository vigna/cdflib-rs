/*
 * Generator for phase-3 reference tables: gamma function family.
 *
 * Writes (a, x, p, q) for gamma_inc, plus (a, ln_gamma) and (a, gamma)
 * tables. The (a, x) grid is chosen to exercise every regime of
 * gamma_inc: small a (power series), moderate a in body and tail
 * (continued fraction, asymptotic expansion), and large a (Temme).
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
    /* ----------------------------------------------------- gamma_log */
    FILE* f = open_csv("tests/data/gamma_log.csv", "a, ln_gamma(a)");
    for (double a = 0.05; a <= 20.0 + 1e-12; a += 0.05) {
        double aa = a;
        fprintf(f, "%.17g,%.17g\n", a, gamma_log(&aa));
    }
    fclose(f);

    /* ------------------------------------------------------- gamma_x */
    f = open_csv("tests/data/gamma_x.csv", "a, gamma(a)");
    for (double a = 0.1; a <= 14.0 + 1e-12; a += 0.1) {
        double aa = a;
        fprintf(f, "%.17g,%.17g\n", a, gamma_x(&aa));
    }
    fclose(f);

    /* ----------------------------------------------------- gamma_inc */
    f = open_csv("tests/data/gamma_inc.csv", "a, x, P(a, x), Q(a, x)");
    /* Cover small, moderate, and large a; for each, sweep x from near 0
       to several times a (covers body, peak, and right tail of P). */
    double as[] = {
        0.25, 0.5, 0.9, 1.0, 1.5, 2.5, 5.0, 9.5, 12.0, 17.0, 25.0,
        50.0, 100.0, 500.0
    };
    for (size_t i = 0; i < sizeof(as) / sizeof(as[0]); i++) {
        double a = as[i];
        double max_x = a > 1.0 ? (3.0 * a) : 8.0;
        double step = max_x / 50.0;
        for (double x = step / 2.0; x <= max_x + 1e-12; x += step) {
            double aa = a, xx = x;
            double p = 0.0, q = 0.0;
            int ind = 0;
            gamma_inc(&aa, &xx, &p, &q, &ind);
            if (p == 2.0) continue; /* error sentinel */
            fprintf(f, "%.17g,%.17g,%.17g,%.17g\n", a, x, p, q);
        }
    }
    fclose(f);

    fprintf(stderr, "wrote 3 tables under tests/data/\n");
    return 0;
}
