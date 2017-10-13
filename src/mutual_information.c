/*
  Mutual information calculation
  Copyright (C) 2017  Bob Mottram <bob@freedombone.net>

  Sugiyama, M., Borgwardt, K.M.
  Measuring Statistical Dependence via the Mutual Information Dimension,
  Proceedings of the 23rd International Joint Conference on
  Artificial Intelligence (IJCAI 2013), Beijing, China, Aug., 2013.
  http://www.ijcai.org/Proceedings/13/Papers/251.pdf

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  3. Neither the name of the University nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.
  .
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE HOLDERS OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "mutual_information.h"

static void discretize(double x[], int codes[], int length, int k)
{
    COUNTDOWN(i, length) {
        codes[i] = floor(x[i] * k);

        if (codes[i] == k)
            codes[i] = k - 1;
    }
}

static double entropy_single(long int B[], int length, int m)
{
    double p, result = 0;

    COUNTUP(i, m) {
        if ((B[i] != length) && (B[i] != 0)) {
            p = B[i] / (double)length;
            result += -1 * p * (log(p) / log(2));
        }
    }

    return result;
}

static double entropy_covariance(long int **B, int length, int m)
{
    double p, result = 0;

    COUNTUP(i, m) {
        COUNTUP(j, m) {
            if ((B[i][j] != length) && (B[i][j] != 0)) {
                p = B[i][j] / (double)length;
                result += -1 * p * (log(p) / log(2));
            }
        }
    }

    return result;
}

static void linear_regression(int length, double x[], double y[],
                              double *a, double *b, double *rsq)
{
    double error = 0, tot = 0, xave = 0, yave = 0, xvar = 0, xyvar = 0;

    COUNTDOWN(i, length) {
        xave += x[i];
        yave += y[i];
    }
    xave = xave / length;
    yave = yave / length;

    COUNTDOWN(i, length) {
        xvar += pow(xave - x[i], 2);
        tot += pow(yave - y[i], 2);
        xyvar += (xave - x[i]) * (yave - y[i]);
    }
    xvar = xvar / length;
    xyvar = xyvar / length;

    *a = xyvar / xvar;
    *b = yave - *a * xave;

    COUNTDOWN(i, length) {
        error += pow((*a * x[i] + *b) - y[i], 2);
    }

    *rsq = 1 - error / tot;
}

static double estimate(int xnum, double yall[], int width,
                       _Bool cov, double minent)
{
    _Bool flag;
    int i_end;
    double a, b, rsq, rsq_before = 0, coef = 0, *x, *y;

    i_end = xnum - width + 1;

    x = (double *)malloc(sizeof(double) * width);
    if (!x) {
        printf("Error allocating memory for estimate x array\n");
        return 0;
    }

    y = (double *)malloc(sizeof(double) * width);
    if (!y) {
        printf("Error allocating memory for estimate y array\n");
        return 0;
    }

    FOR(i, 1, i_end+1) {
        flag = 0;

        COUNTUP(j, width) {
            x[j] = i + j;
            y[j] = yall[i + j - 1];
            if (j > 0) {
                if (y[j] == y[j - 1]) {
                    flag = 1;
                }
            }
        }

        if (flag == 1) {
            a = 0; rsq = 0;
        } else {
            linear_regression(width, x, y, &a, &b, &rsq);
        }

        if (cov == 0) {
            if (rsq > rsq_before) coef = a;
        } else {
            if (rsq > rsq_before && a > minent) coef = a;
        }

        rsq_before = rsq;
    }

    if ((cov == 1) && (coef == 0))
        coef = minent;

    return coef;
}

static double estimate_covariance(int xnum, double yall[], int width,
                                  _Bool cov, double minent)
{
    double res;

    do {
        if (width > 1) {
            res = estimate(xnum, yall, width--, cov, minent);
        } else {
            res = 0;
            break;
        }
    } while (res <= 0);

    return res;
}

static double keepmax(double x, double y, double xy)
{
    double tmp = x > y ? x : y;
    return (xy > tmp ? xy : tmp);
}

static double information_dimension(double x[], double y[], int length,
                                    int level_max, int level_max_cov)
{
    int k, m = 0, width, *codes_x, *codes_y;
    long int *B_x, *B_y, **B_xy;
    double *result_x, *result_y, *result_xy;
    double idim_x = 0, idim_y = 0, idim_xy = 0;

    codes_x = (int *)malloc(sizeof(int) * length);
    if (!codes_x) {
        printf("Error allocating codes_x array\n");
        return 0;
    }

    codes_y = (int *)malloc(sizeof(int) * length);
    if (!codes_y) {
        printf("Error allocating codes_y array\n");
        return 0;
    }

    result_x = (double *)malloc(sizeof(double) * level_max);
    if (!result_x) {
        printf("Error allocating result_x array\n");
        return 0;
    }

    result_y = (double *)malloc(sizeof(double) * level_max);
    if (!result_y) {
        printf("Error allocating result_y array\n");
        return 0;
    }

    result_xy = (double *)malloc(sizeof(double) * level_max_cov);
    if (!result_xy) {
        printf("Error allocating result_xy array\n");
        return 0;
    }

    FOR(level, 1, level_max+1) {
        k = 1 << level;

        B_x = (long int *)malloc(sizeof(long int) * k);
        if (!B_x) {
            printf("Error allocating B_x array\n");
            return 0;
        }

        B_y = (long int *)malloc(sizeof(long int) * k);
        if (!B_y) {
            printf("Error allocating B_y array\n");
            return 0;
        }

        memset((void*)B_x, '\0', sizeof(long int) * k);
        memset((void*)B_y, '\0', sizeof(long int) * k);

        if (level <= level_max_cov) {
            B_xy = (long int **)malloc(sizeof(long int *) * k);
            if (!B_xy) {
                printf("Error allocating B_xy array\n");
                return 0;
            }

            COUNTDOWN(i, k) {
                B_xy[i] = (long int *)malloc(sizeof(long int) * k);
                if (!B_xy[i]) {
                    printf("Error allocating B_xy[%d] array\n", i);
                    return 0;
                }

                memset((void*)B_xy[i], '\0', sizeof(long int) * k);
            }
        }

        discretize(x, codes_x, length, k);
        discretize(y, codes_y, length, k);

        COUNTDOWN(i, length) {
            (B_x[codes_x[i]])++;
            (B_y[codes_y[i]])++;
            if (level <= level_max_cov)
                (B_xy[codes_x[i]][codes_y[i]])++;
        }

        result_x[m] = entropy_single(B_x, length, k);
        result_y[m] = entropy_single(B_y, length, k);

        if (level <= level_max_cov)
            result_xy[m] = entropy_covariance(B_xy, length, k);

        m++;

        free(B_x);
        free(B_y);

        if (level <= level_max_cov) {
            COUNTDOWN(i, k) {
                free(B_xy[i]);
            }
            free(B_xy);
        }
    }

    free(codes_x);
    free(codes_y);

    width = ceil(log(length) / log(4));
    idim_x = estimate(level_max, result_x, width, 0, 0);
    idim_y = estimate(level_max, result_y, width, 0, 0);
    idim_xy = estimate_covariance(level_max_cov, result_xy, width, 1,
                                   idim_x < idim_y ? idim_x : idim_y);
    idim_xy = keepmax(idim_x, idim_y, idim_xy);

    free(result_x);
    free(result_y);
    free(result_xy);
    return idim_xy;
}

/**
 * @brief Returns the mutual information value between two arrays
 *        of a given length. Array values should be in the 0.0 -> 1.0 range
 * @param x First array of values in the range 0.0 -> 1.0
 * @param y Second array of values in the range 0.0 -> 1.0
 * @param length Length of the arrays
 * @returns Mutual information value
 */
double mutual_information(double x[], double y[], int length)
{
    int level_max = floor(log(length) / log(2));
    int level_max_cov = floor(log(length) / log(4)) + 4;

    /* sanitize the arrays */
    COUNTDOWN(i, length) {
        if (x[i] < 0) x[i] = 0;
        if (x[i] > 1) x[i] = 1;
        if (y[i] < 0) y[i] = 0;
        if (y[i] > 1) y[i] = 1;
        if (isnan(x[i])) x[i] = 0;
        if (isnan(y[i])) y[i] = 0;
    }

    return information_dimension(x, y, length, level_max, level_max_cov);
}
