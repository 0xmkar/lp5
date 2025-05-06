#include <iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;

// Fixed the kernel declaration with proper double underscores
__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// Added error checking helper function
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N;
    cout << "Enter the size of the vectors: ";
    cin >> N;
    
    // Validate input
    if (N <= 0) {
        cerr << "Error: Vector size must be positive" << endl;
        return EXIT_FAILURE;
    }
    
    vector<int> A(N), B(N), C(N);
    
    cout << "Enter " << N << " elements for vector A:\n";
    for (int i = 0; i < N; ++i) cin >> A[i];
    
    cout << "Enter " << N << " elements for vector B:\n";
    for (int i = 0; i < N; ++i) cin >> B[i];
    
    int *d_A, *d_B, *d_C;
    
    // Added error checking for CUDA operations
    checkCudaError(cudaMalloc(&d_A, N * sizeof(int)), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc(&d_B, N * sizeof(int)), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc(&d_C, N * sizeof(int)), "Failed to allocate device memory for C");
    
    checkCudaError(cudaMemcpy(d_A, A.data(), N * sizeof(int), cudaMemcpyHostToDevice), 
                  "Failed to copy A from host to device");
    checkCudaError(cudaMemcpy(d_B, B.data(), N * sizeof(int), cudaMemcpyHostToDevice),
                  "Failed to copy B from host to device");
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    checkCudaError(cudaEventRecord(start), "Failed to record start event");
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to get elapsed time");
    
    checkCudaError(cudaMemcpy(C.data(), d_C, N * sizeof(int), cudaMemcpyDeviceToHost),
                  "Failed to copy C from device to host");
    
    cout << "Result (first 10 elements): ";
    for (int i = 0; i < min(10, N); ++i) cout << C[i] << " ";
    cout << "\nElapsed Time: " << milliseconds << " ms\n";
    
    // Free resources
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    
    return 0;
}