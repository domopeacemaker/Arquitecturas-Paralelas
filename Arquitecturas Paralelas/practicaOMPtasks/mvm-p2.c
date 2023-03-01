#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N 32768

int main(int argc, char **argv) 
{
  int i, j;
  double **mat, bvec[N], cvec[N];
  double start, end;

  mat = (double **) malloc(N * sizeof(double *));
  mat[0] = (double *) malloc(N * N * sizeof(double));
  for (i=1; i<N; i++)
    mat[i] = mat[0] + i*N;
  
  /*--------------------------------------------------------------------------*/

  printf("----- Initialization of matrix and vector ------\n");

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      mat[i][j] = (double) i+j+1;
    }
  }

  for (i = 0; i < N; i++) {
    bvec[i] = (double) i+1;
  }

  for (i = 0; i < N; i++) {
    cvec[i] = 0.0;
  }

  printf("----- End of Initialization ------\n\n");

  /*--------------------------------------------------------------------------*/

  printf("----- Matrix Vector Multiplication ------\n");

  start = omp_get_wtime();
  #pragma omp parallel for private(j) schedule(dynamic,16)
  for (i = 0; i < N; i++) {
    for (j = i; j < N; j++)
      cvec[i] += mat[i][j] * bvec[j];
  }
  end = omp_get_wtime();

  printf("mat[0][0]=%lf, mat[%d][%d]=%lf\n", mat[0][0], N-1, N-1, mat[N-1][N-1]);
  printf("bvec[0]=%lf, bvec[%d]=%lf\n", bvec[0], N-1, bvec[N-1]);
  printf("cvec[0]=%lf, cvec[%d]=%lf\n", cvec[0], N-1, cvec[N-1]);

  printf("----- End of Matrix Multiplication ------\n\n");
  printf("Time: %f\n", end-start);
}
