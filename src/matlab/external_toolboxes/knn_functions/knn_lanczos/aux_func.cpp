#include "aux_func.h"

#include <time.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;


void generate_rand_X(double **X, int n, int D)
{
  int i, j;

  srand((unsigned)time(NULL));

  for (i = 0; i < n; i++)
    for (j = 0; j < D; j++)
      X[i][j] = double(rand())/RAND_MAX;
}


// data in file was stored in binary format
bool read_X_from_file(double **X, int n, int D, char *filename)
{
  //printf("reading in data set %s...... ", filename);
  //fflush(stdout);

  FILE *fp = NULL;
  if (!(fp = fopen(filename, "rb")))
    return false;

  int i;
  int num_in, num_total;
  for (i = 0; i < n; i++)
    {
      num_total = 0;
      while (num_total < D)
        {
          num_in = fread(X[i]+num_total, sizeof(double), D, fp);
          num_total += num_in;
        }
    }

  fclose(fp);

  //printf("done\n");

  return true;
}


// compute the squared distance between two D dimensional vector
double comp_dist_sq(double *x, double *y, int D)
{
  int i;
  double dist_sq = 0, tmp;
  double *x_ptr = x;
  double *y_ptr = y;

  for (i = 0; i < D; i++)
    {
      tmp = x[i] - y[i];
      dist_sq += tmp * tmp;
    }

  return dist_sq;
}


double comp_knn_accuracy(int **accurate_knn, int **approx_knn, int n, int k)
{
  int i, idx_accurate, idx_approx, cnt = 0;

  for (i = 0; i < n; i++)
    {
      idx_accurate = 0;
      idx_approx = 0;
      while (idx_accurate < k)
        {
          while ((approx_knn[i][idx_approx] != accurate_knn[i][idx_accurate])
                 && (idx_accurate < k))
            idx_accurate++;
          if (idx_accurate < k)
            cnt++;
          idx_approx++;
        }
    }

  return ((double)cnt)/(n*k);
}


double comp_ave_rank(double **dist_sq, int n, int k, int **neighbors)
{
  int i, j;
  int rank_sum = 0;
  int *rank_list = new int [n];

  for (i = 0; i < n; i++)
    {
      sort_all_elements(dist_sq[i], n, rank_list);
      for (j = 0; j < k; j++)
        rank_sum += rank_list[ neighbors[i][j] ];
    }

  delete [] rank_list;
  return (double)rank_sum/n/k;
}


bool compar_func(ELE a, ELE b)
{
  return (a.val < b.val);
}


// get the smallest k elements from a length-n array A
// the returned elements are sorted
void smallest_k_elements(double *A, int n, int k, int *idx,
                         SORT_METHOD method)
{
  int i;

  ELE *A_copy = new ELE [n];
  for (i = 0; i < n; i++)
    {
      A_copy[i].val = A[i];
      A_copy[i].idx = i;
    }

  // two ways to get the sorted smallest k elements
  if (method == PARTIAL_SORT)
    partial_sort(A_copy, A_copy + k, A_copy + n, compar_func);
  else /* method == NTH_ELEMENT_AND_SORT */
    {
      nth_element(A_copy, A_copy + k, A_copy + n, compar_func);
      sort(A_copy, A_copy + k, compar_func);
    }

  for (i = 0; i < k; i++)
    idx[i] = A_copy[i].idx;

  delete [] A_copy;
}


void sort_all_elements(double *A, int n, int *rank_list)
{
  int i;

  ELE *A_copy = new ELE [n];
  for (i = 0; i < n; i++)
    {
      A_copy[i].val = A[i];
      A_copy[i].idx = i;
    }

  sort(A_copy, A_copy + n, compar_func);

  for (i = 0; i < n; i++)
    rank_list[ A_copy[i].idx ] = i+1;

  delete [] A_copy;
}
