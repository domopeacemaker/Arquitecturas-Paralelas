#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef N
#define N 20 
#endif

struct node {
   int index;
   double pi;
   struct node* next;
};

double pic(int nsteps) {
   double step, sum, x, pi;
   int i;

   step = 1.0/(double) (nsteps*1000000);
   sum = 0.0;

   for (i=0; i<nsteps*1000000; i++)
   {
      x = (i-0.5)*step;
      sum = sum + 4.0/(1.0+x*x);
   }

   pi = step*sum;

   return pi;
}

void processwork(struct node* p) 
{
   int n;

   n = p->index;
   p->pi = pic(n);
}

struct node* init_list(struct node* p) {
    int i;
    struct node* head = NULL;
    struct node* temp = NULL;
    
    head = malloc(sizeof(struct node));
    p = head;
    p->index = 1;
    p->pi = 0;
    for (i=2; i<=N; i++) {
       temp  =  malloc(sizeof(struct node));
       p->next = temp;
       p = temp;
       p->index = i;
       p->pi = 0;
    }
    p->next = NULL;
    return head;
}

int main() 
{
     double start, end;
     struct node *p=NULL;
     struct node *temp=NULL;
     struct node *head=NULL;
     
     printf("Process linked list\n");
     printf("  The linked list (ll) will be initialized to %d nodes\n",N);
     printf("  The ll node with index 'k' will compute the PI number with 'k'*1000000 steps, where k=1,...,N\n\n");

     p = init_list(p);
     head = p;

     start = omp_get_wtime();

     while (p != NULL) {
        processwork(p);
        p = p->next;
     }

     end = omp_get_wtime();

     printf("Computed pi's:\n");
     p = head;
     while (p != NULL) {
        printf("   pi[%d] = %.12f\n", p->index, p->pi);
        temp = p->next;
        free (p);
        p = temp;
     }  
     free (p);

     printf("Compute Time: %f seconds\n", end - start);
}
