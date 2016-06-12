#ifndef _LINEAR_ALGEBRA_H_
#define _LINEAR_ALGEBRA_H_

void   lan_svd1(double **A, double *a, double *b, int m, int n,
                double *u, double *s, double *v);
void   lan_svd1_ATA(double **A, double *a, double *b, int m, int n, int k,
                    double *u, double *s, double *v);
void   lan_svd1_AAT(double **A, double *a, double *b, int m, int n, int k,
                    double *u, double *s, double *v);

void   blas_a(double a, double *y, int n);
double blas_dot(double *x, double *y, int n);
void   blas_ax(double a, double *x, double *y, int n);
void   blas_saxpy(double a, double *x, double *y, int n);
void   blas_mat_vec(double **A, double *x, double *y, int m, int n);
void   blas_matT_vec(double **A, double *x, double *y, int m, int n);

#endif
