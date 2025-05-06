#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Fixed kernel declaration with proper double underscores
__global__ void matrixMul(int* A, int* B, int* C, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N && c < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[r * N + k] * B[k * N + c];
        C[r * N + c] = sum;
    }
}

// Error checking helper function
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N;
    cout << "Enter matrix size N (NxN): ";
    cin >> N;
    
    // Validate input
    if (N <= 0) {
        cerr << "Error: Matrix size must be positive" << endl;
        return EXIT_FAILURE;
    }
    
    int size = N * N * sizeof(int);
    int *A = new int[N * N], *B = new int[N * N], *C = new int[N * N];
    
    cout << "Enter matrix A (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << "A[" << i << "][" << j << "]: ";
            cin >> A[i * N + j];
        }
    }
    
    cout << "Enter matrix B (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << "B[" << i << "][" << j << "]: ";
            cin >> B[i * N + j];
        }
    }
    
    int *dA, *dB, *dC;
    checkCudaError(cudaMalloc(&dA, size), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc(&dB, size), "Failed to allocate device memory for B"); 
    checkCudaError(cudaMalloc(&dC, size), "Failed to allocate device memory for C");
    
    checkCudaError(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice), 
                  "Failed to copy A from host to device");
    checkCudaError(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice),
                  "Failed to copy B from host to device");
    
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    // Added timing measurement
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    checkCudaError(cudaEventRecord(start), "Failed to record start event");
    
    matrixMul<<<blocks, threads>>>(dA, dB, dC, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");
    
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to get elapsed time");
    
    checkCudaError(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost),
                  "Failed to copy C from device to host");
    
    cout << "\nResult Matrix C:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << C[i * N + j] << " ";
        cout << "\n";
    }
    
    cout << "\nMatrix multiplication completed in " << milliseconds << " ms\n";
    
    // Free resources
    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    delete[] A; 
    delete[] B; 
    delete[] C;
    
    return 0;
}