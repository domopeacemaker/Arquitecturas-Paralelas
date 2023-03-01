#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048

double gettime(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(int argc, char *argv[])
{
    // float a[N][N], b[N][N], c[N][N];
    float **a, **b, **c;
    int i,j,k;
    float sum = 0.0;
    double dtime;

    // Allocate matrices

    a = (float **) malloc (N*sizeof(float **));
    posix_memalign((void **)&a[0], 16, N*N*sizeof(float **));
    for (i=1; i<N; i++) a[i] = a[i-1] + N;
    b = (float **) malloc (N*sizeof(float **));
    posix_memalign((void **)&b[0], 16, N*N*sizeof(float **));
    for (i=1; i<N; i++) b[i] = b[i-1] + N;
    c = (float **) malloc (N*sizeof(float **));
    posix_memalign((void **)&c[0], 16, N*N*sizeof(float **));
    for (i=1; i<N; i++) c[i] = c[i-1] + N;

    // Initialize input matrices
        #pragma omp simd
    for (i = 0; i < N; i++)
       for (j = 0; j < N; j++)
       {
          a[i][j] = b[i][j] = 3.14;
          c[i][j] = 0.0;
       }

    // Matrix multiplication
    dtime = gettime();

        #pragma omp parallel for simd
    for (i = 0; i < N; i++)
       for (k = 0; k < N; k++)
          for (j = 0; j < N; j++)
          c[i][j] = c[i][j] + a[i][k]*b[k][j];

dtime = gettime() - dtime;

for (i = 0; i < N; i++)
for (j = 0; j < N; j++)
sum += c[i][j];

// Print results
printf("Sum for matrix C: %12.4f\n",sum);
printf("Exec time: %9.5f sec.\n",dtime);
}
