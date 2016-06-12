#ifndef _KNN_ALGO_H_
#define _KNN_ALGO_H_

#include "aux_func.h"
#include "dynamic_array.h"

void kNN_bruteforce(double **X, int n, int D, int k,
                    int **neighbors, double **knn_dist_sq);
void kNN_bruteforce2(double **X, int n, int D, int k,
                     double **dist_sq, int **neighbors, double **knn_dist_sq);
void kNN_w_hash_table(double **X, int D, int k, int *label, int num,
                      int **neighbors, DynamicArray *dist_sq);
void kNN_dNc_disjoint(double **X, int D, int k, int *label, int num, int p,
                      int **neighbors, DynamicArray *dist_sq);
void kNN_dNc_overlap(double **X, int D, int k, int *label, int num,
                     int p, double r, bool is_refine,
                     int **neighbors, DynamicArray *dist_sq,
                     int *num_need_dist, int *num_comp_dist);
void kNN_dNc_glue(double **X, int D, int k, int *label, int num,
                  int p, double r, bool is_refine,
                  int **neighbors, DynamicArray *dist_sq,
                  int *num_need_dist, int *num_comp_dist);

void comp_sep_plane(double **X, int n, int D,
                    double *pln_normal, double *pln_dist);

void partition_disjoint(double *pln_dist, int n,
                        int * &left, int *nl, int * &right, int *nr);
void partition_overlap(double *pln_dist, int n, double r,
                       int * &left, int *nl, int * &right, int *nr);
void partition_glue(double *pln_dist, int n, double r,
                    int * &left, int *nl, int * &right, int *nr,
                    int * &middle, int *nm);

void refine_knn(double **X, int D, int k, int *label, int num,
                int **neighbors, DynamicArray *dist_sq,
                int *num_need_dist, int *num_comp_dist);

#endif
