#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define N 16777216

float A[N];

double gettime(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(int argc, char *argv[]) 
{
    int i;
    float sum = 0.0;
    double dtime;

    // Initialize input matrix
    for (i = 0; i < N; i++)
       A[i] = 3.14;

    // Processing
    dtime = gettime();

#pragma omp simd safelen(8)
    for (i = 8; i < N; i++) 
       A[i] = A[i] + A[i-8];

    dtime = gettime() - dtime;

    for (i = 0; i < N; i++)
       sum += A[i];

    // Print results
    printf("Sum for matrix A: %12.4f\n",sum);
    printf("Exec time: %9.5f sec.\n",dtime); 
}
