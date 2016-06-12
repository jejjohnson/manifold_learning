#include "dynamic_array.h"

#include <math.h>
#include <float.h>
#include <string.h>


DynamicArray::DynamicArray(int size)
{
  m_Size = size;
  m_Idx = new int [m_Size];
  m_Val = new double [m_Size];
  m_NumElements = 0;
}

DynamicArray::~DynamicArray()
{
  if (m_Idx)
    delete [] m_Idx;
  if (m_Val)
    delete [] m_Val;
}

int DynamicArray::hasElement(int idx)
{
  int i;
  for (i = 0; i < m_NumElements; i++)
    if (m_Idx[i] == idx)
      return i;
  return -1;
}

void DynamicArray::addElement(int idx, double val)
{
  if (m_NumElements == m_Size)
    {
      int m_Size_new = m_Size * 2;

      int *m_Idx_new = new int [m_Size_new];
      memcpy(m_Idx_new, m_Idx, m_Size*sizeof(int));
      delete [] m_Idx;
      m_Idx = m_Idx_new;

      double *m_Val_new = new double [m_Size_new];
      memcpy(m_Val_new, m_Val, m_Size*sizeof(double));
      delete [] m_Val;
      m_Val = m_Val_new;

      m_Size = m_Size_new;
    }
  
  m_Idx[m_NumElements] = idx;
  m_Val[m_NumElements] = val;
  m_NumElements++;
}

void DynamicArray::compSmallestElements(int k, int *idx, double *val,
                                        SORT_METHOD method)
{
  int i;
  int k_tmp = (k < m_NumElements) ? k : m_NumElements;
  int *idx_tmp = new int [k_tmp];
  smallest_k_elements(m_Val, m_NumElements, k_tmp, idx_tmp, method);

  if (idx)
    {
      for (i = 0; i < k_tmp; i++)
        idx[i] = m_Idx[idx_tmp[i]];
      for (i = k_tmp; i < k; i++)
        idx[i] = -1;
    }

  if (val)
    {
      for (i = 0; i < k_tmp; i++)
        val[i] = m_Val[idx_tmp[i]];
      for (i = k_tmp; i < k; i++)
        val[i] = DBL_MAX;
    }
}


bool output_knn(int n, int k, int **neighbors, DynamicArray *dist_sq,
                char *filename)
{
  //printf("outputing kNN graph ...... ");
  //fflush(stdout);

  FILE *fp = NULL;
  if (!(fp = fopen(filename, "w")))
    return false;

  int i, j, jj;
  int loc;
  double val;
  for (i = 0; i < n; i++)
    for (j = 0; j < k; j++)
      {
        jj = neighbors[i][j];
        if (jj < 0)
          {
            //printf("neighbors[%d][%d] = %d\n", i, j, jj);
            continue;
          }
        loc = dist_sq[i].hasElement(jj);
        if (loc != -1)
          val = dist_sq[i].getElementVal(loc);
        else
          perror("output_knn(): an unexpected distance");
        fprintf(fp, "%7d %7d  %g\n", jj+1, i+1, sqrt(val));
      }

  fclose(fp);
  //printf("done\n");
  return true;
}


bool output_knn(int n, int k, int **neighbors, double **dist_sq,
                char *filename)
{
  //printf("outputing kNN graph ...... ");
  //fflush(stdout);

  FILE *fp = NULL;
  if (!(fp = fopen(filename, "w")))
    return false;

  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < k; j++)
      fprintf(fp,"%7d %7d  %g\n", neighbors[i][j]+1, i+1, sqrt(dist_sq[i][j]));

  fclose(fp);
  //printf("done\n");
  return true;
}


double comp_dist_percent(DynamicArray *dist_sq, int n)
{
  int i, total = 0;
  for (i = 0; i < n; i++)
    total += dist_sq[i].getNumElements();
  return (double)total/(n*(n-1));
}
