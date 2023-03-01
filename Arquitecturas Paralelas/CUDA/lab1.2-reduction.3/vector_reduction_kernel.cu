#define NUM_ELEMENTS 512


// CUDA kernel to perform the reduction in parallel on the GPU
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
__global__ void reduction(float *g_data, int n)
{
  int stride;
  // Define shared memory
  __shared__ float scratch[NUM_ELEMENTS];

 // Load the shared memory
  scratch[threadIdx.x] = g_data[threadIdx.x];
  if(threadIdx.x + blockDim.x < n){
    scratch[blockDim.x + threadIdx.x] = g_data[ blockDim.x + threadIdx.x];

  }
  __syncthreads();

 /* ESQUEMA 1 */

/*
 for( stride=NUM_ELEMENTS/2; stride>=1; stride = stride/2 ) 
  {
      if ( threadIdx.x<stride)
         scratch[threadIdx.x ] += scratch[stride + threadIdx.x ];
      __syncthreads();
  }
*/


/*ESQUEMA 2*/


 for(stride=1; stride<n; stride*=2) 
  {

    if (threadIdx.x % stride==0)
         scratch[2*threadIdx.x ] += scratch[2*threadIdx.x+stride];

      __syncthreads();
  }



/*ESQUEMA 3*/
/*
 for(stride=NUM_ELEMENTS/2;stride>=1; stride=stride/2) 
 {

   if (threadIdx.x <= stride)
         scratch[threadIdx.x] += scratch[2*stride-(threadIdx.x+1)];

      __syncthreads();
  }
*/

 // Store results back to global memory
 if(threadIdx.x == 0)
    g_data[0] = scratch[0];

  return;
}
