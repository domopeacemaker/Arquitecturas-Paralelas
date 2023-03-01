#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) 
{
#pragma omp parallel
{
  printf("A ");
  printf("race ");
  printf("car ");
}

  printf("\n");
}
