#ifndef _AUX_FUNC_H_
#define _AUX_FUNC_H_

#include <stdio.h>


typedef struct tagELE
{
  double val;
  int idx;
} ELE;

typedef enum tagSORT_METHOD
{
  PARTIAL_SORT,
  NTH_ELEMENT_AND_SORT
} SORT_METHOD;


//template <typename T> void new_2d_array(T ** &A, int dim1, int dim2);
//template <typename T> void delete_2d_array(T **A, int dim1, int dim2);
void generate_rand_X(double **X, int n, int D);
bool read_X_from_file(double **X, int n, int D, char *filename);
double comp_dist_sq(double *x, double *y, int D);
double comp_knn_accuracy(int **accurate_knn, int **approx_knn, int n, int k);
double comp_ave_rank(double **dist_sq, int n, int k, int **neighbors);
bool compar_func(ELE a, ELE b);
void smallest_k_elements(double *A, int n, int k, int *idx,
                         SORT_METHOD method);
void sort_all_elements(double *A, int n, int *rank_list);


template <typename T> void new_2d_array(T ** &A, int dim1, int dim2)
{
  int i;
  A = new T * [dim1];
  if (!A)
    perror("new_2d_array(): fail to allocate memory");

  for (i = 0; i < dim1; i++)
    {
      A[i] = NULL;
      A[i] = new T [dim2];
      if (!A[i])
        perror("new_2d_array(): fail to allocate memory");
    }
}

template <typename T> void delete_2d_array(T **A, int dim1, int dim2)
{
  int i;
  if (A)
    {
      for (i = 0; i < dim1; i++)
        {
          if (A[i])
            delete [] A[i];
        }
      delete [] A;
      A = NULL;
    }
}

#endif
