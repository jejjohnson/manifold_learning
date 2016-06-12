#ifndef _DYNAMIC_ARRAY_H_
#define _DYNAMIC_ARRAY_H_

#include "aux_func.h"


class DynamicArray
{
 public:
  DynamicArray(int size = 100);
  ~DynamicArray();

  int hasElement(int idx);
  void addElement(int idx, double val);
  void compSmallestElements(int k, int *idx, double *val, SORT_METHOD method);

  double getElementVal(int loc) { return m_Val[loc]; }
  int getNumElements(void) { return m_NumElements; }

 private:
  int m_Size;
  int m_NumElements;
  int *m_Idx;
  double *m_Val;
};


bool output_knn(int n, int k, int **neighbors, DynamicArray *dist_sq,
                char *filename);
bool output_knn(int n, int k, int **neighbors, double **dist_sq,
                char *filename);
double comp_dist_percent(DynamicArray *dist_sq, int n);

#endif
