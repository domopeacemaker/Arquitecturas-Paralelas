#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double gettime(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

void sinx (int N, int terms, float *x, float *result)
{
  int i,j;

  for (i=0; i<N; i++)
  {
    float value = x[i];
    float numer = x[i]*x[i]*x[i];
    int denom = 6;
    int sign = -1;

    for (j=1; j<=terms; j++)
    {
      value += sign*numer/denom;
      numer *= x[i]*x[i];
      denom *= (2*j+2)*(2*j+3);
      sign *= -1;
    }

    result[i] = value;
  }
}

int main(int argc, char *argv[])
{
  int N, terms;
  float *x, *result;
  int i;
  float sum = 0.0;
  double dtime;

  if (argc > 2){
    N = atoi(argv[1]);
    terms = atoi(argv[2]);
  } else if (argc > 1){
    N = atoi(argv[1]);
    terms = 12;
  } else{
    N = 67108864;
    terms = 12;
  }
  printf("Using N=%d and terms=%d\n",N,terms);

  x = (float *) malloc(N*sizeof(float));
  posix_memalign((void *)&x, 16, N*sizeof(float *));
  result = (float *) calloc(N,sizeof(float));
  posix_memalign((void *)&result, 16, N*sizeof(float *));

  for (i=0; i<N; i++)
    x[i] = 1.0;

  dtime = gettime();

  sinx(N,terms,x,result);

  dtime = gettime() - dtime;

  for (i=0; i<N; i++)
    sum += result[i];

  printf("Sum for result: %12.4f\n",sum);
  printf("Exec time: %9.5f sec.\n",dtime); 
}
