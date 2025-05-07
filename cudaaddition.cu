#include <iostream>  // For input/output operations
#include <vector>    // For storing the vectors on the host
#include <cuda_runtime.h>  // For CUDA runtime API functions
using namespace std;

// CUDA kernel function that runs on the GPU
// __global__ indicates this function runs on the device (GPU) and is called from host code
__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    // Calculate the global thread ID
    // threadIdx.x is the thread index within a block
    // blockIdx.x is the block index within the grid
    // blockDim.x is the number of threads per block
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Ensure we don't access memory beyond the vector size
    if (idx < N) 
        C[idx] = A[idx] + B[idx];  // Each thread computes one element of C
}

// Helper function to check for CUDA errors
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        // Print error message and error string if a CUDA operation fails
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);  // Exit the program with failure status
    }
}

int main() {
    int N;  // Size of the vectors
    cout << "Enter the size of the vectors: ";
    cin >> N;
    
    // Input validation
    if (N <= 0) {
        cerr << "Error: Vector size must be positive" << endl;
        return EXIT_FAILURE;
    }
    
    // Host vectors
    vector<int> A(N), B(N), C(N);
    
    // Input for vector A
    cout << "Enter " << N << " elements for vector A:\n";
    for (int i = 0; i < N; ++i) cin >> A[i];
    
    // Input for vector B
    cout << "Enter " << N << " elements for vector B:\n";
    for (int i = 0; i < N; ++i) cin >> B[i];
    
    // Device (GPU) pointers
    int *d_A, *d_B, *d_C;
    
    // Allocate memory on the GPU for each vector
    checkCudaError(cudaMalloc(&d_A, N * sizeof(int)), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc(&d_B, N * sizeof(int)), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc(&d_C, N * sizeof(int)), "Failed to allocate device memory for C");
    
    // Copy host vectors to device memory
    checkCudaError(cudaMemcpy(d_A, A.data(), N * sizeof(int), cudaMemcpyHostToDevice), 
                  "Failed to copy A from host to device");
    checkCudaError(cudaMemcpy(d_B, B.data(), N * sizeof(int), cudaMemcpyHostToDevice),
                  "Failed to copy B from host to device");
    
    // Define the execution configuration
    int threadsPerBlock = 256;  // Number of threads per block
    // Calculate number of blocks needed, rounding up to ensure all elements are processed
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    // Record the start event
    checkCudaError(cudaEventRecord(start), "Failed to record start event");
    
    // Launch the CUDA kernel
    // <<<blocksPerGrid, threadsPerBlock>>> is the execution configuration
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Record the stop event
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    // Wait for the stop event to complete
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    // Calculate the elapsed time between start and stop events
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to get elapsed time");
    
    // Copy the result from device to host
    checkCudaError(cudaMemcpy(C.data(), d_C, N * sizeof(int), cudaMemcpyDeviceToHost),
                  "Failed to copy C from device to host");
    
    // Display the result (up to 10 elements)
    cout << "Result (first 10 elements): ";
    for (int i = 0; i < min(10, N); ++i) cout << C[i] << " ";
    cout << "\nElapsed Time: " << milliseconds << " ms\n";
    
    // Free GPU memory
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    
    // Destroy CUDA events
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    
    return 0;
}

// Sample input and expected flow:
// Enter the size of the vectors: 5
// Enter 5 elements for vector A:
// 1 2 3 4 5
// Enter 5 elements for vector B:
// 5 4 3 2 1
// 
// Result (first 10 elements): 6 6 6 6 6 
// Elapsed Time: 0.012345 ms