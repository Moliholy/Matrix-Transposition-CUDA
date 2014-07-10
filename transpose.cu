#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>


#define BLOCK 16


__global__
void parallelTransposeMemCoalescing(int* A, int* B, int m, int n) {
        __shared__ int block[BLOCK][BLOCK];

        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m && j < n) {
                block[threadIdx.y][threadIdx.x] = A[i * n + j];
                __syncthreads();
                B[j * m + i] = block[threadIdx.y][threadIdx.x];
        }
}


__global__
void parallelTranspose(int* A, int* B, int m, int n) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m && j < n) {
                B[j * m + i] = A[i * n + j];
        }
}

int main(int argc, char *argv[]) {
        if (argc < 3) {
                printf("No enough arguments.");
                return -1;
        }

        srand(time(NULL));

        int m = atoi(argv[1]);
        int n = atoi(argv[2]);

        int* A = (int*) malloc(m * n * sizeof(int));
        int* B = (int*) malloc(m * n * sizeof(int));

        int i;
        for (i = 0; i < m * n; ++i)
                A[i] = rand() % 10;


        int *d_A, *d_B;
        cudaMalloc(&d_A, n * m * sizeof(int));
        cudaMalloc(&d_B, n * m * sizeof(int));

        //dimensions
        dim3 threadblock(BLOCK, BLOCK);
        dim3 grid(1 + n / threadblock.x, 1 + m / threadblock.y);

        //copying A to the GPU
        cudaMemcpy(d_A, A, n * m * sizeof(int), cudaMemcpyHostToDevice);





        /////////////////////////// FIRST EXECUTION ///////////////////////////
        clock_t t = clock();

        //calling function
        parallelTranspose<<<grid, threadblock>>>(d_A, d_B, m, n);
        cudaDeviceSynchronize();

        //once the function has been called I copy the result in matrix
        cudaMemcpy(B, d_B, n * m * sizeof(int), cudaMemcpyDeviceToHost);

        double parallelExecutionTime = ((double) (clock() - t))
                        / ((double) (CLOCKS_PER_SEC));




        /////////////////////////// SECOND EXECUTION ///////////////////////////
        t = clock();

        //calling function
        parallelTransposeMemCoalescing<<<grid, threadblock>>>(d_A, d_B, m, n);
        cudaDeviceSynchronize();

        //once the function has been called I copy the result in matrix
        cudaMemcpy(B, d_B, n * m * sizeof(int), cudaMemcpyDeviceToHost);

        double improvedParallelExecutionTime = ((double) (clock() - t))
                        / ((double) (CLOCKS_PER_SEC));





        /////////////////////////// PRINTING RESULTS ///////////////////////////

        printf("%d;%f;%f\n", m, parallelExecutionTime, improvedParallelExecutionTime);

        cudaFree(d_A);
        cudaFree(d_B);

        free(A);
        free(B);

        return 0;
}
