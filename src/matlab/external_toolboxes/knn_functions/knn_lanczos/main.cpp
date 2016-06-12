#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "aux_func.h"
#include "knn_algo.h"
#include "dynamic_array.h"


int main(void)
{
  // parameters
  int n = 1965;                 // # data points
  int D = 560;                  // data dimension
  char fname[] = "example.dat"; // data file
  char outf[] = "example.knn";  // output knn graph to file
  double r = 0.3;               // overlap size
  int k = 12;                   // # neighbors
  int p = k*5;                  // largest size for bruteforce

  // read in data
  double **X = NULL;
  new_2d_array(X, n, D);
  if (!read_X_from_file(X, n, D, fname))
    {
      printf("error open file.\n");
      exit(1);
    }


  /*
  //////////////////////
  //  bruteforce kNN  //
  //////////////////////
  // stores all the nearest neighbors
  int **neighbors = NULL;
  new_2d_array(neighbors, n, k);
  // stores all the knn squared distances
  double **knn_dist_sq = NULL;
  new_2d_array(knn_dist_sq, n, n);

  // the main driver
  clock_t start = clock();
  kNN_bruteforce(X, n, D, k, neighbors, knn_dist_sq);
  clock_t end = clock();

  // you may want to output the knn graph to file
  if (!output_knn(n, k, neighbors, knn_dist_sq, outf))
    printf("error open output file.\n");

  // release memory
  delete_2d_array(neighbors, n, k);
  delete_2d_array(knn_dist_sq, n, k);

  // print elapsed time
  double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("elapsed time: %.2f\n", cpu_time_used);
  */


  ////////////////////////////////////////////
  //  divide and conquer kNN, glue version  //
  ////////////////////////////////////////////
  // stores all the nearest neighbors
  int **neighbors = NULL;
  new_2d_array(neighbors, n, k);
  // stores all the knn squared distances
  DynamicArray *knn_dist_sq = new DynamicArray [n];
  // the auxilary label array
  int *label = new int [n];
  int i;
  for (i = 0; i < n; i++)
    label[i] = i;
  // some variables you don't care about
  int num_need_dist, num_comp_dist;

  // the main driver
  clock_t start = clock();
  kNN_dNc_glue(X, D, k, label, n, p, r, true, neighbors, knn_dist_sq,
               &num_need_dist, &num_comp_dist);
  clock_t end = clock();

  // you may want to output the knn graph to file
  if (!output_knn(n, k, neighbors, knn_dist_sq, outf))
    printf("error open output file.\n");

  // release memory
  delete_2d_array(neighbors, n, k);
  delete [] knn_dist_sq;

  // print elapsed time
  double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("elapsed time: %.2f\n", cpu_time_used);


  /*
  ///////////////////////////////////////////////
  //  divide and conquer kNN, overlap version  //
  ///////////////////////////////////////////////
  // stores all the nearest neighbors
  int **neighbors = NULL;
  new_2d_array(neighbors, n, k);
  // stores all the knn squared distances
  DynamicArray *knn_dist_sq = new DynamicArray [n];
  // the auxilary label array
  int *label = new int [n];
  int i;
  for (i = 0; i < n; i++)
    label[i] = i;
  // some variables you don't care about
  int num_need_dist, num_comp_dist;

  // the main driver
  clock_t start = clock();
  kNN_dNc_overlap(X, D, k, label, n, p, r, true, neighbors, knn_dist_sq,
                  &num_need_dist, &num_comp_dist);
  clock_t end = clock();

  // you may want to output the knn graph to file
  if (!output_knn(n, k, neighbors, knn_dist_sq, outf))
    printf("error open output file.\n");

  // release memory
  delete_2d_array(neighbors, n, k);
  delete [] knn_dist_sq;

  // print elapsed time
  double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("elapsed time: %.2f\n", cpu_time_used);
  */


  // clean up
  delete_2d_array(X, n, D);
  return 0;
}
