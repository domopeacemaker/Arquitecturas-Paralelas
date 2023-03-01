#include <stdio.h>
#include <omp.h>

int a = 1;

void main()
{
   int b = 2, c = 3;
   #pragma omp parallel shared(b) private(c)
   {
      int d = 4;
      #pragma omp task
      {
         int e = 5;
         // a is shared
         // b is shared
         // c is firstprivate
         // d is firstprivate
         // e is private
	 printf("a: %d; b: %d\n", a, b);
	 printf("c: %d; d: %d\n", c, d);
	 printf("e: %d\n", e);
      }
   }
}
