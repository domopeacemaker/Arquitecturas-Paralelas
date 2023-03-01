#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <smmintrin.h>

double gettime(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

/*********** Original
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
***********/

/*********** Desenrrollado
void sinx (int N, int terms, float *x, float *result)
{
  int i,j;

  for (i=0; i<N; i+=4)
  {
    float value1 = x[i]; float value2 = x[i+1]; float value3 = x[i+2]; float value4 = x[i+3];
    float numer1 = x[i]*x[i]*x[i]; float numer2 = x[i+1]*x[i]*x[i+1]; 
    float numer3 = x[i+2]*x[i]*x[i+2]; float numer4 = x[i+3]*x[i]*x[i+3];
    int denom1 = 6; int denom2 = 6; int denom3 = 6; int denom4 = 6;
    int sign = -1;

    for (j=1; j<=terms; j++)
    {
      value1 += sign*numer1/denom1; value2 += sign*numer2/denom2; 
      value3 += sign*numer3/denom3; value4 += sign*numer4/denom4;
      numer1 *= x[i]*x[i]; numer2 *= x[i+1]*x[i+1];
      numer3 *= x[i+2]*x[i+2]; numer4 *= x[i+3]*x[i+3];
      denom1 *= (2*j+2)*(2*j+3); denom2 *= (2*j+2)*(2*j+3); denom3 *= (2*j+2)*(2*j+3); denom4 *= (2*j+2)*(2*j+3);
      sign *= -1;
    }

    result[i] = value1;
    result[i+1] = value2;
    result[i+2] = value3;
    result[i+3] = value4;
  }
}
***********/

void sinx (int N, int terms, float *x, float *result)
{
  int i,j;

  for (i=0; i<N; i+=4)
  {
    __m128 origx = _mm_load_ps(&x[i]);
    __m128 value = origx;
    //float value = x[i];
    __m128 numer = _mm_mul_ps(origx,_mm_mul_ps(origx,origx));
    //float numer = x[i]*x[i]*x[i];
    __m128 denom = _mm_set1_ps((float) 6);
    //int denom = 6;
    int sign = -1;

    for (j=1; j<=terms; j++)
    {
      __m128 tmp = _mm_div_ps(_mm_mul_ps(_mm_set1_ps((float) sign),numer),denom);
      value = _mm_add_ps(value,tmp);
      //value += sign*numer/denom;
      numer = _mm_mul_ps(numer,_mm_mul_ps(origx,origx));
      //numer *= x[i]*x[i];
      denom = _mm_mul_ps(denom,_mm_set1_ps((float)(2*j+2)*(2*j+3)));
      //denom *= (2*j+2)*(2*j+3);
      sign *= -1;
    }

    _mm_store_ps(&result[i],value);
    //result[i] = value;
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
