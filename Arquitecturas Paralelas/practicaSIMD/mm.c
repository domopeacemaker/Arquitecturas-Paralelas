#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048

float a[N][N], b[N][N], c[N][N];

int main(int argc, char *argv[]) 
{
    int i,j,k;
    struct timeval t1, t2;

    // Initialize input matrices
    for (i = 0; i < N; i++)
       for (j = 0; j < N; j++)
       {
          a[i][j] = b[i][j] = 3.14;
          c[i][j] = 0.0;
       }

    // Matrix multiplication
    gettimeofday(&t1,NULL);

    for (i = 0; i < N; i++) 
       for (k = 0; k < N; k++) 
          for (j = 0; j < N; j++) 
             c[i][j] = c[i][j] + a[i][k]*b[k][j];

    gettimeofday(&t2,NULL);


    // Print results
    printf("a b c: %f %f %f\n",a[0][0],b[0][0],c[0][0]);
    printf("Exec time: %f sec.\n",(t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/(double)1000000); 
}
