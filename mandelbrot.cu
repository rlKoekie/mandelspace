
#include <stdio.h>
#include <cuda.h>

extern "C" {
   void mandelbrot_row_calc(int *nStep_ptr, float *deltaStep_ptr, float *start_x_ptr, float *start_y_ptr, int *mandelSize_ptr, int *innerLoopSize_ptr, int *outerLoopSize_ptr, unsigned int *results);
   void mini_mandelbrot_calc(float *start_x_ptr, float *start_y_ptr, int *mandelSize_ptr, int *innerLoopSize_ptr, int *outerLoopSize_ptr, unsigned int *results);
   void PrintLastCUDAError();
}


__global__ void mini_mandel_kernel(unsigned int *dev_Results, float start_x, float start_y, int innerloopsize, int outerloopsize) {
   int bx = blockIdx.x;
   int by = blockIdx.y;
   int mandsize = gridDim.x;

   int i,j;
   unsigned int threadscore = 0;

   float xpos = (float)-1.0 + (float)4.0*float(bx - mandsize/2 )/(float)mandsize;
   float ypos = (float)0.0 + (float)4.0*float(by - mandsize/2 )/(float)mandsize;

   float a = start_x;
   float a_tmp;
   float b = start_y;

   for(i=0;i<outerloopsize;i++){
       for(j=0;j<innerloopsize;j++){
           a_tmp = a;
           a = a*a - b*b + xpos;
           b = (float)2*a_tmp*b + ypos;
       }
       if ((a*a + b*b) <= (float)4.0){
           threadscore += i;
       } else {
           break;
       }
   }
   dev_Results[by*mandsize + bx] = threadscore;
   return;
}


void mini_mandelbrot_calc(float *start_x_ptr, float *start_y_ptr, int *mandelSize_ptr, int *innerLoopSize_ptr, int *outerLoopSize_ptr, unsigned int *results) {
   float start_x = *start_x_ptr;
   float start_y = *start_y_ptr;
   int mandelSize = *mandelSize_ptr;
   int innerloopsize = *innerLoopSize_ptr;
   int outerloopsize = *outerLoopSize_ptr;

   unsigned int *dev_Results;
   cudaMalloc( (void**)&dev_Results, mandelSize * mandelSize * sizeof(unsigned int) );

   // define the launch dimensions
   dim3    threads(1);
   dim3    grid(mandelSize,mandelSize);

   // run the kernel
   mini_mandel_kernel<<<grid,threads>>>(dev_Results, start_x, start_y, innerloopsize, outerloopsize);
   cudaMemcpy( results, dev_Results, mandelSize * mandelSize * sizeof(unsigned int), cudaMemcpyDeviceToHost );
   cudaFree( dev_Results );
   return;
}


__global__ void row_calc(unsigned int *dev_Results, float deltaStep, float start_x, float start_y, int innerloopsize, int outerloopsize){
//    __shared__ bool killus;
   __shared__ unsigned int threadscores[16][16];
   __syncthreads();
//    killus = 0;
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int mandsize = blockDim.x;

   int i,j;
   unsigned int threadscore = 0;

   float xpos = (float)-1.0 + (float)4.0*float(tx - mandsize/2 )/(float)mandsize;
   float ypos = (float)0.0 + (float)4.0*float(ty - mandsize/2 )/(float)mandsize;

   float a = start_x + deltaStep*(float)bx;
   float a_tmp;
   float b = start_y;

   for(i=0;i<outerloopsize;i++){
       for(j=0;j<innerloopsize;j++){
           a_tmp = a;
           a = a*a - b*b + xpos;
           b = (float)2*a_tmp*b + ypos;
       }
       if ((a*a + b*b) <= (float)4.0){
           threadscore += i;
       }
   }
   threadscores[tx][ty] = threadscore;
   __syncthreads();

   if (ty == 0){ // the first thread of the row will sum the row
       threadscore = 0;
       for (i=0;i<mandsize;i++){
           threadscore += threadscores[tx][i];
       } // and put it in the first slot of the row in shared mem
       threadscores[tx][0] = threadscore;
   }
   __syncthreads();

   if (tx == 0 && ty == 0 ){ // tread (0,0) will do the sum over the
       threadscore = 0; // first column, to get the total
       for (i=0;i<mandsize;i++){
           threadscore +=threadscores[i][0];
       }
       dev_Results[bx] = threadscore; // the result to global_mem
   }
   return;
}



// Call this to print out the CUDA error status. To check if GPU stuff worked correctly,
// this should be called after a synchronous function (like a normal memory copy).
void PrintLastCUDAError(){
   cudaError_t err = cudaGetLastError();
   printf(cudaGetErrorString( err ));
   printf("\n");
   return;
}

void mandelbrot_row_calc(int *nStep_ptr, float *deltaStep_ptr, float *start_x_ptr, float *start_y_ptr, int *mandelSize_ptr, int *innerLoopSize_ptr, int *outerLoopSize_ptr, unsigned int *results) {
   int nStep = *nStep_ptr;
   int mandelSize = *mandelSize_ptr;
   float deltaStep = *deltaStep_ptr;
   float start_x = *start_x_ptr;
   float start_y = *start_y_ptr;
   int innerloopsize = *innerLoopSize_ptr;
   int outerloopsize = *outerLoopSize_ptr;

   unsigned int *dev_Results;
   cudaMalloc( (void**)&dev_Results, nStep * sizeof(unsigned int) );
   cudaMemset( dev_Results, 0, nStep * sizeof(unsigned int));

//    PrintLastCUDAError();

           // define the launch dimensions
   dim3    threads(mandelSize,mandelSize); // size of the cuda-blocks
   dim3    grid(nStep);

   // run the kernel
   row_calc<<<grid,threads>>>(dev_Results, deltaStep, start_x, start_y, innerloopsize, outerloopsize);
   cudaMemcpy( results, dev_Results, nStep * sizeof(unsigned int), cudaMemcpyDeviceToHost );
   cudaFree( dev_Results );
//    PrintLastCUDAError();
   return;
}

