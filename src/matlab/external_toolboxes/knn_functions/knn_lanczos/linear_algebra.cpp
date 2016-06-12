#include "linear_algebra.h"

#include <time.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "aux_func.h"


// This function computes the largest singular triplet of the rank-1
// correction matrix A-a*b. The notions here conform to the typical
// numerical linear algebra notations:
//   A: m-by-n    a: m-by-1    b: 1-by-n
//   u: m-by-1    s: 1-by-1    v: n-by-1
void lan_svd1(double **A, double *a, double *b, int m, int n,
              double *u, double *s, double *v)
{
  int nsteps = 5;
  if (nsteps > m)
    nsteps = m;
  if (nsteps > n)
    nsteps = n;

  if (m >= n)
    lan_svd1_ATA(A, a, b, m, n, nsteps, u, s, v);
  else
    lan_svd1_AAT(A, a, b, m, n, nsteps, u, s, v);
}


void lan_svd1_ATA(double **A, double *a, double *b, int m, int n, int k,
                  double *u, double *s, double *v)
{
  int i, j;
  double c;

  // allocate memory
  double **q = NULL;
  new_2d_array(q, k+2, n);
  /* each q[i] corresponds to a lanczos vector. */
  /* the 2d array q is indeed the transpose of the Q matrix. */
  double *alpha = new double [k+1];
  double *beta = new double [k+2];
  double *w = new double [n];
  double *z = new double [m];
  
  // initialize beta_1, q_0, q_1
  beta[1] = 0;
  blas_a(0, q[0], n);
  //blas_a(0, q[1], n);
  //q[1][0] = 1;
  srand((unsigned)time(NULL));
  for (i = 0; i < n; i++)
    q[1][i] = (double)(rand());
  double sum_sq = 0;
  for (i = 0; i < n; i++)
    sum_sq += q[1][i] * q[1][i];
  double norm = sqrt(sum_sq);
  for (i = 0; i < n; i++)
    q[1][i] /= norm;

  // main loop
  for (i = 1; i <= k; i++)
    {
      // c <- b*q_i
      c = blas_dot(b, q[i], n);

      // z <- A*q_i
      blas_mat_vec(A, q[i], z, m, n);

      // z <- (A-a*b)*q_i = z-c*a
      blas_saxpy(-c, a, z, m);

      // c <- a'*z
      c = blas_dot(a, z, m);

      // w <- A'*z
      blas_matT_vec(A, z, w, m, n);

      // w <- (A-a*b)'*(A-a*b)*q_i = (A-a*b)'*z = A'*z-c*b' = w-c*b'
      blas_saxpy(-c, b, w, n);

      // w <- (A-a*b)'*(A-a*b)*q_i-beta_i*q_{i-1} = w-beta_i*q_{i-1}
      blas_saxpy(-beta[i], q[i-1], w, n);

      // alpha_i <- w'*q_i
      alpha[i] = blas_dot(w, q[i], n);

      // w <- w - alpha_i*q_i
      blas_saxpy(-alpha[i], q[i], w, n);

      // w <- w - sum((w'*q_j)*q_j)
      for (j = 1; j <= i-1; j++)
        {
          c = blas_dot(w, q[j], n);
          blas_saxpy(-c, q[j], w, n);
        }

      // beta_{i+1} <- norm(w)
      beta[i+1] = sqrt(blas_dot(w, w, n));

      // q_{i+1} <- w/beta_{i+1}
      blas_ax(1./beta[i+1], w, q[i+1], n);
    }

  // construct T and compute its eigen-elements
  double *T = new double [k*k];
  memset(T, 0, k*k*sizeof(double));
  for (i = 0; i < k; i++)
    T[i*k+i] = alpha[i+1];
  for (i = 0; i < k-1; i++)
    T[i*k+i+1] = beta[i+2];
  for (i = 0; i < k-1; i++)
    T[(i+1)*k+i] = beta[i+2];
     
  gsl_matrix_view mv = gsl_matrix_view_array(T, k, k);
  gsl_vector *eval = gsl_vector_alloc(k);
  gsl_matrix *evec = gsl_matrix_alloc(k, k);
  gsl_eigen_symmv_workspace *wsp = gsl_eigen_symmv_alloc(k);

  gsl_eigen_symmv(&mv.matrix, eval, evec, wsp);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);

  // s <- sqrt of largest eigenvalue of T
  (*s) = sqrt(gsl_vector_get(eval, 0));

  // v <- Q*v0
  gsl_vector_view vv0 = gsl_matrix_column(evec, 0);
  double *v0 = new double [k];
  for (i = 0; i < k; i++)
    v0[i] = gsl_vector_get(&(vv0.vector), i);
  double **QT = NULL;
  new_2d_array(QT, k, n);
  for (i = 0; i < k; i++)
    memcpy(QT[i], q[i+1], n*sizeof(double));
  blas_matT_vec(QT, v0, v, k, n);

  // u <- (A-a*b)*v/s
  c = blas_dot(b, v, n);
  blas_mat_vec(A, v, u, m, n);
  blas_saxpy(-c, a, u, m);
  blas_ax(1./(*s), u, u, m);

  // release memory
  delete_2d_array(q, k+2, n);
  delete_2d_array(QT, k, n);
  delete [] alpha;
  delete [] beta;
  delete [] w;
  delete [] z;
  delete [] T;
  delete [] v0;

  gsl_eigen_symmv_free(wsp);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}


void lan_svd1_AAT(double **A, double *a, double *b, int m, int n, int k,
                  double *u, double *s, double *v)
{
  int i, j;
  double c;

  // allocate memory
  double **q = NULL;
  new_2d_array(q, k+2, m);
  /* each q[i] corresponds to a lanczos vector. */
  /* the 2d array q is indeed the transpose of the Q matrix. */
  double *alpha = new double [k+1];
  double *beta = new double [k+2];
  double *w = new double [m];
  double *z = new double [n];
  
  // initialize beta_1, q_0, q_1
  beta[1] = 0;
  blas_a(0, q[0], m);
  //blas_a(0, q[1], m);
  //q[1][0] = 1;
  srand((unsigned)time(NULL));
  for (i = 0; i < m; i++)
    q[1][i] = (double)(rand());
  double sum_sq = 0;
  for (i = 0; i < m; i++)
    sum_sq += q[1][i] * q[1][i];
  double norm = sqrt(sum_sq);
  for (i = 0; i < m; i++)
    q[1][i] /= norm;

  // main loop
  for (i = 1; i <= k; i++)
    {
      // c <- a'*q_i
      c = blas_dot(a, q[i], m);

      // z <- A'*q_i
      blas_matT_vec(A, q[i], z, m, n);

      // z <- (A-a*b)'*q_i = z-c*b'
      blas_saxpy(-c, b, z, n);

      // c <- b*z
      c = blas_dot(b, z, n);

      // w <- A*z
      blas_mat_vec(A, z, w, m, n);

      // w <- (A-a*b)*(A-a*b)'*q_i = (A-a*b)*z = A*z-c*a = w-c*a
      blas_saxpy(-c, a, w, m);

      // w <- (A-a*b)*(A-a*b)'*q_i-beta_i*q_{i-1} = w-beta_i*q_{i-1}
      blas_saxpy(-beta[i], q[i-1], w, m);

      // alpha_i <- w'*q_i
      alpha[i] = blas_dot(w, q[i], m);

      // w <- w - alpha_i*q_i
      blas_saxpy(-alpha[i], q[i], w, m);

      // w <- w - sum((w'*q_j)*q_j)
      for (j = 1; j <= i-1; j++)
        {
          c = blas_dot(w, q[j], m);
          blas_saxpy(-c, q[j], w, m);
        }

      // beta_{i+1} <- norm(w)
      beta[i+1] = sqrt(blas_dot(w, w, m));

      // q_{i+1} <- w/beta_{i+1}
      blas_ax(1./beta[i+1], w, q[i+1], m);
    }

  // construct T and compute its eigen-elements
  double *T = new double [k*k];
  memset(T, 0, k*k*sizeof(double));
  for (i = 0; i < k; i++)
    T[i*k+i] = alpha[i+1];
  for (i = 0; i < k-1; i++)
    T[i*k+i+1] = beta[i+2];
  for (i = 0; i < k-1; i++)
    T[(i+1)*k+i] = beta[i+2];

  gsl_matrix_view mv = gsl_matrix_view_array(T, k, k);
  gsl_vector *eval = gsl_vector_alloc(k);
  gsl_matrix *evec = gsl_matrix_alloc(k, k);
  gsl_eigen_symmv_workspace *wsp = gsl_eigen_symmv_alloc(k);

  gsl_eigen_symmv(&mv.matrix, eval, evec, wsp);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);

  // s <- sqrt of largest eigenvalue of T
  (*s) = sqrt(gsl_vector_get(eval, 0));

  // u <- Q*v0
  gsl_vector_view vv0 = gsl_matrix_column(evec, 0);
  double *v0 = new double [k];
  for (i = 0; i < k; i++)
    v0[i] = gsl_vector_get(&(vv0.vector), i);
  double **QT = NULL;
  new_2d_array(QT, k, m);
  for (i = 0; i < k; i++)
    memcpy(QT[i], q[i+1], m*sizeof(double));
  blas_matT_vec(QT, v0, u, k, m);

  // v <- (A-a*b)'*u/s
  c = blas_dot(a, u, m);
  blas_matT_vec(A, u, v, m, n);
  blas_saxpy(-c, b, v, n);
  blas_ax(1./(*s), v, v, n);

  // release memory
  delete_2d_array(q, k+2, m);
  delete_2d_array(QT, k, m);
  delete [] alpha;
  delete [] beta;
  delete [] w;
  delete [] z;
  delete [] T;
  delete [] v0;

  gsl_eigen_symmv_free(wsp);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}


// y = a
void blas_a(double a, double *y, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = a;
}


// c = x'*y
double blas_dot(double *x, double *y, int n)
{
  int i;
  double c = 0;
  for (i = 0; i < n; i++)
    c += x[i]*y[i];
  return c;
}


// y = a*x
void blas_ax(double a, double *x, double *y, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = a*x[i];
}


// y = a*x+y
void blas_saxpy(double a, double *x, double *y, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] += a*x[i];
}


// y = A*x
void blas_mat_vec(double **A, double *x, double *y, int m, int n)
{
  int i, j;
  for (i = 0; i < m; i++)
    {
      y[i] = 0;
      for (j = 0; j < n; j++)
        y[i] += A[i][j] * x[j];
    }
}


// y = A'*x
void blas_matT_vec(double **A, double *x, double *y, int m, int n)
{
  int i, j;
  for (j = 0; j < n; j++)
    y[j] = 0;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      y[j] += A[i][j] * x[i];
}
