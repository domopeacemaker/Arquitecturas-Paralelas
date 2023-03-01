#include <stdio.h>

int main() 
{
   int x = 1;

   #pragma omp parallel
   #pragma omp single
   {
      #pragma omp task shared(x)
         x = 10; 
      #pragma omp task shared(x)
         printf("x + 1 = %d\n", x+1);
      #pragma omp task shared(x)
         printf("x + 2 = %d\n", x+2);
      #pragma omp task shared(x)
         printf("x + 3 = %d\n", x+3);
      #pragma omp task shared(x)
         printf("x + 4 = %d\n", x+4);
   }
}
