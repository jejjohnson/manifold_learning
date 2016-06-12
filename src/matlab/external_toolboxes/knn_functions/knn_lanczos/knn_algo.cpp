#include "knn_algo.h"

#include <math.h>
#include <float.h>
#include <string.h>
#include <algorithm>
#include "linear_algebra.h"

#define M_METHOD PARTIAL_SORT

using namespace std;


// require: n > k
void kNN_bruteforce(double **X, int n, int D, int k,
                    int **neighbors, double **knn_dist_sq)
{
  int i, j;

  // allocate 2d array to store square distances
  double **dist_sq = NULL;
  new_2d_array(dist_sq, n, n);

  // compute pairwise square distances
  for (i = 0; i < n; i++)
    dist_sq[i][i] = DBL_MAX;

  for (i = 0; i < n; i++)
    for (j = i+1; j < n; j++)
      {
        dist_sq[i][j] = comp_dist_sq(X[i], X[j], D);
        dist_sq[j][i] = dist_sq[i][j];
      }

  // for each data point, find k nearest neighbors
  // the neighbors are sorted according to distances
  for (i = 0; i < n; i++)
    smallest_k_elements(dist_sq[i], n, k, neighbors[i], M_METHOD);

  for (i = 0; i < n; i++)
    for (j = 0; j < k; j++)
      knn_dist_sq[i][j] = dist_sq[i][neighbors[i][j]];

  // release memory
  delete_2d_array(dist_sq, n, n);
}


void kNN_bruteforce2(double **X, int n, int D, int k,
                     double **dist_sq, int **neighbors, double **knn_dist_sq)
{
  int i, j;

  // compute pairwise square distances
  for (i = 0; i < n; i++)
    dist_sq[i][i] = DBL_MAX;

  for (i = 0; i < n; i++)
    for (j = i+1; j < n; j++)
      {
        dist_sq[i][j] = comp_dist_sq(X[i], X[j], D);
        dist_sq[j][i] = dist_sq[i][j];
      }

  // for each data point, find k nearest neighbors
  // the neighbors are sorted according to distances
  for (i = 0; i < n; i++)
    smallest_k_elements(dist_sq[i], n, k, neighbors[i], M_METHOD);

  for (i = 0; i < n; i++)
    for (j = 0; j < k; j++)
      knn_dist_sq[i][j] = dist_sq[i][neighbors[i][j]];
}


// implicitly be able to handle num <= k case
void kNN_w_hash_table(double **X, int D, int k, int *label, int num,
                      int **neighbors, DynamicArray *dist_sq)
{
  int i, j, ii, jj;

  double dist_sq_tmp;
  for (i = 0; i < num; i++)
    for (j = i+1; j < num; j++)
      {
        ii = label[i];
        jj = label[j];
        if (dist_sq[ii].hasElement(jj) == -1)
          {
            dist_sq_tmp = comp_dist_sq(X[ii], X[jj], D);
            dist_sq[ii].addElement(jj, dist_sq_tmp);
            dist_sq[jj].addElement(ii, dist_sq_tmp);
          }
      }

  for (i = 0; i < num; i++)
    {
      ii = label[i];
      dist_sq[ii].compSmallestElements(k, neighbors[ii], NULL, M_METHOD);
    }
}


void kNN_dNc_disjoint(double **X, int D, int k, int *label, int num, int p,
                      int **neighbors, DynamicArray *dist_sq)
{
  int i;

  // divide step: partition the set X(label)
  int *left = NULL, *right = NULL;
  int nl, nr;
  double *pln_normal = new double [D];
  double *pln_dist = new double [num];
  double **Y = NULL;
  new_2d_array(Y, num, D);
  for (i = 0; i < num; i++)
    memcpy(Y[i], X[label[i]], D*sizeof(double));
  comp_sep_plane(Y, num, D, pln_normal, pln_dist);
  partition_disjoint(pln_dist, num, left, &nl, right, &nr);

  //printf("kNN_dNc_disjoint: num = %d, nl = %d, nr = %d\n", num, nl, nr);

  // recursively compute kNN for the left side
  int *label_l = new int [nl];
  for (i = 0; i < nl; i++)
    label_l[i] = label[left[i]];
  if (nl <= p)
    kNN_w_hash_table(X, D, k, label_l, nl, neighbors, dist_sq);
  else
    kNN_dNc_disjoint(X, D, k, label_l, nl, p, neighbors, dist_sq);

  // recursively compute kNN for the right side
  int *label_r = new int [nr];
  for (i = 0; i < nr; i++)
    label_r[i] = label[right[i]];
  if (nr <= p)
    kNN_w_hash_table(X, D, k, label_r, nr, neighbors, dist_sq);
  else
    kNN_dNc_disjoint(X, D, k, label_r, nr, p, neighbors, dist_sq);

  // conquer step: conquer the results from both sides.
  // nothing needs to be done.

  // release memory
  delete [] pln_normal;
  delete [] pln_dist;
  delete [] left;
  delete [] right;
  delete [] label_l;
  delete [] label_r;
  delete_2d_array(Y, num, D);
}


void kNN_dNc_overlap(double **X, int D, int k, int *label, int num,
                     int p, double r, bool is_refine,
                     int **neighbors, DynamicArray *dist_sq,
                     int *num_need_dist, int *num_comp_dist)
{
  int i;

  // divide step: partition the set X(label)
  int *left = NULL, *right = NULL;
  int nl, nr;
  double *pln_normal = new double [D];
  double *pln_dist = new double [num];
  double **Y = NULL;
  new_2d_array(Y, num, D);
  for (i = 0; i < num; i++)
    memcpy(Y[i], X[label[i]], D*sizeof(double));
  comp_sep_plane(Y, num, D, pln_normal, pln_dist);
  partition_overlap(pln_dist, num, r, left, &nl, right, &nr);

  //printf("kNN_dNc_overlap: num = %d, nl = %d, nr = %d\n", num, nl, nr);

  // recursively compute kNN for the left side
  int *label_l = new int [nl];
  for (i = 0; i < nl; i++)
    label_l[i] = label[left[i]];
  if (nl <= p)
    kNN_w_hash_table(X, D, k, label_l, nl, neighbors, dist_sq);
  else
    kNN_dNc_overlap(X, D, k, label_l, nl, p, r, is_refine, neighbors, dist_sq,
                    num_need_dist, num_comp_dist);

  // recursively compute kNN for the right side
  int *label_r = new int [nr];
  for (i = 0; i < nr; i++)
    label_r[i] = label[right[i]];
  if (nr <= p)
    kNN_w_hash_table(X, D, k, label_r, nr, neighbors, dist_sq);
  else
    kNN_dNc_overlap(X, D, k, label_r, nr, p, r, is_refine, neighbors, dist_sq,
                    num_need_dist, num_comp_dist);

  // conquer step: conquer the results from both sides.
  // implicitly done in kNN_w_hash_table.

  // do refinement
  if (is_refine)
    {
      int n_need_dist, n_comp_dist;
      refine_knn(X, D, k, label, num, neighbors, dist_sq,
                 &n_need_dist, &n_comp_dist);
      *num_need_dist += n_need_dist;
      *num_comp_dist += n_comp_dist;
    }

  // release memory
  delete [] pln_normal;
  delete [] pln_dist;
  delete [] left;
  delete [] right;
  delete [] label_l;
  delete [] label_r;
  delete_2d_array(Y, num, D);
}


void kNN_dNc_glue(double **X, int D, int k, int *label, int num,
                  int p, double r, bool is_refine,
                  int **neighbors, DynamicArray *dist_sq,
                  int *num_need_dist, int *num_comp_dist)
{
  int i;

  // divide step: partition the set X(label)
  int *left = NULL, *right = NULL, *middle = NULL;
  int nl, nr, nm;
  double *pln_normal = new double [D];
  double *pln_dist = new double [num];
  double **Y = NULL;
  new_2d_array(Y, num, D);
  for (i = 0; i < num; i++)
    memcpy(Y[i], X[label[i]], D*sizeof(double));
  comp_sep_plane(Y, num, D, pln_normal, pln_dist);
  partition_glue(pln_dist, num, r, left, &nl, right, &nr, middle, &nm);

  //printf("kNN_dNc_glue: num = %d, nl = %d, nr = %d, nm = %d\n", num, nl, nr, nm);

  // recursively compute kNN for the left part
  int *label_l = new int [nl];
  for (i = 0; i < nl; i++)
    label_l[i] = label[left[i]];
  if (nl <= p)
    kNN_w_hash_table(X, D, k, label_l, nl, neighbors, dist_sq);
  else
    kNN_dNc_glue(X, D, k, label_l, nl, p, r, is_refine, neighbors, dist_sq,
                 num_need_dist, num_comp_dist);

  // recursively compute kNN for the right part
  int *label_r = new int [nr];
  for (i = 0; i < nr; i++)
    label_r[i] = label[right[i]];
  if (nr <= p)
    kNN_w_hash_table(X, D, k, label_r, nr, neighbors, dist_sq);
  else
    kNN_dNc_glue(X, D, k, label_r, nr, p, r, is_refine, neighbors, dist_sq,
                 num_need_dist, num_comp_dist);

  // recursively compute kNN for the middle part
  int *label_m = new int [nm];
  for (i = 0; i < nm; i++)
    label_m[i] = label[middle[i]];
  if (nm <= p)
    kNN_w_hash_table(X, D, k, label_m, nm, neighbors, dist_sq);
  else
    kNN_dNc_glue(X, D, k, label_m, nm, p, r, is_refine, neighbors, dist_sq,
                 num_need_dist, num_comp_dist);

  // conquer step: conquer the results from all three parts.
  // implicitly done in kNN_w_hash_table.

  // do refinement
  if (is_refine)
    {
      int n_need_dist, n_comp_dist;
      refine_knn(X, D, k, label, num, neighbors, dist_sq,
                 &n_need_dist, &n_comp_dist);
      *num_need_dist += n_need_dist;
      *num_comp_dist += n_comp_dist;
    }

  // release memory
  delete [] pln_normal;
  delete [] pln_dist;
  delete [] left;
  delete [] right;
  delete [] middle;
  delete [] label_l;
  delete [] label_r;
  delete [] label_m;
  delete_2d_array(Y, num, D);
}


void comp_sep_plane(double **X, int n, int D,
                    double *pln_normal, double *pln_dist)
{
  int i, j;

  // compute the mean of the data set X
  double *mean = new double [D];
  memset(mean, 0, D*sizeof(double));
  for (i = 0; i < n; i++)
    for (j = 0; j < D; j++)
      mean[j] += X[i][j];
  for (j = 0; j < D; j++)
    mean[j] /= n;

  // SVD on X-ones*mean; compute the largest triplet (u,s,v)
  // NOTE THE MATRIX DIMENSIONS AND VECTOR LENGTHS
  double *ones = new double [n];
  for (i = 0; i < n; i++)
    ones[i] = 1;
  double s;
  lan_svd1(X, ones, mean, n, D, pln_dist, &s, pln_normal);

  // release memory
  delete [] mean;
  delete [] ones;
}


void partition_disjoint(double *pln_dist, int n,
                        int * &left, int *nl, int * &right, int *nr)
{
  int i;

  (*nl) = 0;
  (*nr) = 0;
  for (i = 0; i < n; i++)
    {
      if (pln_dist[i] < 0)
        (*nl)++;
      else
        (*nr)++;
    }

  left = new int [*nl];
  right = new int [*nr];
  int idx_l = 0;
  int idx_r = 0;
  for (i = 0; i < n; i++)
    {
      if (pln_dist[i] < 0)
        left[idx_l++] = i;
      else
        right[idx_r++] = i;
    }
}


void partition_overlap(double *pln_dist, int n, double r,
                       int * &left, int *nl, int * &right, int *nr)
{
  int i;

  double *array = new double [n];
  for (i = 0; i < n; i++)
    array[i] = fabs(pln_dist[i]);
  int loc = (int)(r*n);
  nth_element(array, array + loc, array + n);
  double left_boarder = array[loc];
  double right_boarder = -array[loc];

  (*nl) = 0;
  (*nr) = 0;
  for (i = 0; i < n; i++)
    {
      if (pln_dist[i] < left_boarder)
        (*nl)++;
      if (pln_dist[i] >= right_boarder)
        (*nr)++;
    }

  left = new int [*nl];
  right = new int [*nr];
  int idx_l = 0;
  int idx_r = 0;
  for (i = 0; i < n; i++)
    {
      if ((pln_dist[i] < left_boarder) && (idx_l < (*nl)))
        left[idx_l++] = i;
      if ((pln_dist[i] >= right_boarder) && (idx_r < (*nr)))
        right[idx_r++] = i;
    }

  delete [] array;
}


void partition_glue(double *pln_dist, int n, double r,
                    int * &left, int *nl, int * &right, int *nr,
                    int * &middle, int *nm)
{
  int i;

  double *array = new double [n];
  for (i = 0; i < n; i++)
    array[i] = fabs(pln_dist[i]);
  int loc = (int)(r*n);
  nth_element(array, array + loc, array + n);
  double left_boarder = array[loc];
  double right_boarder = -array[loc];

  (*nl) = 0;
  (*nr) = 0;
  (*nm) = 0;
  for (i = 0; i < n; i++)
    {
      if (pln_dist[i] < 0)
        (*nl)++;
      else
        (*nr)++;
      if ((pln_dist[i] < left_boarder) && (pln_dist[i] >= right_boarder))
        (*nm)++;
    }

  left = new int [*nl];
  right = new int [*nr];
  middle = new int [*nm];
  int idx_l = 0;
  int idx_r = 0;
  int idx_m = 0;
  for (i = 0; i < n; i++)
    {
      if (pln_dist[i] < 0)
        left[idx_l++] = i;
      else
        right[idx_r++] = i;
      if ((pln_dist[i] < left_boarder) && (pln_dist[i] >= right_boarder)
          && (idx_m < (*nm)))
        middle[idx_m++] = i;
    }

  delete [] array;
}


void refine_knn(double **X, int D, int k, int *label, int num,
                int **neighbors, DynamicArray *dist_sq,
                int *num_need_dist, int *num_comp_dist)
{
  int i, j, ii, jj, z, zz, x;
  int num_elements;
  int loc;
  double val;
  ELE *array = new ELE [k*(k+1)];

  *num_need_dist = 0;
  *num_comp_dist = 0;

  for (i = 0; i < num; i++)
    {
      ii = label[i];
      num_elements = 0;

      // grab the children
      for (j = 0; j < k; j++)
        {
          jj = neighbors[ii][j];
          if (jj < 0)
            break;

          loc = dist_sq[ii].hasElement(jj);
          (*num_need_dist)++;
          if (loc != -1)
            val = dist_sq[ii].getElementVal(loc);
          else
            {
              // should never enter here
              val = DBL_MAX;
              perror("refine_knn(): something wrong with the code");
            }
          array[num_elements].idx = jj;
          array[num_elements].val = val;
          num_elements++;
        }

      // grab the grand-children
      for (j = 0; j < k; j++)
        {
          jj = neighbors[ii][j];
          if (jj < 0)
            break;

          for (z = 0; z < k; z++)
            {
              zz = neighbors[jj][z];
              if (zz < 0)
                break;

              // if the grand-child is itself, omit
              if (zz == ii)
                continue;

              // if its grand-child happens to be its child, omit
              x = 0;
              while ((x < num_elements) && (array[x].idx != zz))
                x++;
              if (x < num_elements)
                continue;

              // grab the grand-child
              loc = dist_sq[ii].hasElement(zz);
              (*num_need_dist)++;
              if (loc != -1)
                val = dist_sq[ii].getElementVal(loc);
              else
                {
                  val = comp_dist_sq(X[ii], X[zz], D);
                  dist_sq[ii].addElement(zz, val);
                  dist_sq[zz].addElement(ii, val);
                  (*num_comp_dist)++;
                }
              array[num_elements].idx = zz;
              array[num_elements].val = val;
              num_elements++;
            }
        }

      // sort
      sort(array, array+num_elements, compar_func);

      // get the nearest neighbors
      if (k <= num_elements)
        {
          for (j = 0; j < k; j++)
            neighbors[ii][j] = array[j].idx;
        }
      else
        {
          for (j = 0; j < num_elements; j++)
            neighbors[ii][j] = array[j].idx;
          for (j = num_elements; j < k; j++)
            neighbors[ii][j] = -1;
        }
    }

  delete [] array;
}

