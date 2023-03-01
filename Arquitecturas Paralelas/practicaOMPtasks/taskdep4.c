#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int sum(int x, int y)
{
   sleep (1);
   return x+y;
}

int square(int x)
{
   sleep (1);
   return x*x;
}

void process(int v[6], int *tot)
{
   #pragma omp parallel 
   {
   #pragma omp single
   {
   #pragma omp task depend(inout:v[0]) depend(in:v[1])
   v[0] = sum(v[0],v[1]);
   #pragma omp task depend(inout:v[4]) 
   v[4] = sum(v[4],1);
   #pragma omp task depend(in:v[0],v[3]) depend(out:v[2])
   v[2] = sum(v[3],v[0]);
   #pragma omp task depend(in:v[2],v[4]) depend(out:v[5])
   v[5] = sum(v[2],v[4]);
   #pragma omp task depend(in:v[0],v[4]) depend(out:v[1])
   v[1] = sum(v[0],v[4]);
   #pragma omp task depend(inout:v[3]) 
   v[3] = sum(v[3],-3);

   for(int i=0; i<6; i++)
   {
      *tot += square(v[i]);
   }
   }
   }
}

int main()
{
   int v[6]={1,2,3,4,5,6};
   int tot=0;
   double start,end;

   start = omp_get_wtime();
   process(v, &tot);
   end = omp_get_wtime();

   printf("tot = %d \n", tot);
   printf("time = %f sec.\n", end-start);
}
