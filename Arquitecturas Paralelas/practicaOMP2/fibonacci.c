#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int fib(int n) 
{
   int x, y;
   if (n < 2) {
      return (n);
   } else {
      x = fib(n - 1);
      y = fib(n - 2);
      return (x + y);
   }
}

int main(int argc, char *argv[]) {
     double start, end;
     int N, fibn;
     
     if (argc != 2)
     {
        printf("Use: %s N\n",argv[0]);
        exit(0);
     }
     N = atoi(argv[1]);

     printf("Compute Fibonacci number %d\n", N);

     start = omp_get_wtime();
     fibn = fib(N);
     end = omp_get_wtime();

     printf("Fibonacci number %d: %d\n", N, fibn);
     printf("Compute Time: %f seconds\n", end - start);

     return 0;
}

