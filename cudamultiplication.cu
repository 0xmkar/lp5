#include <iostream>  // For input/output operations
#include <cuda_runtime.h>  // For CUDA runtime API functions
using namespace std;

// CUDA kernel function for matrix multiplication
// __global__ indicates this function runs on the device (GPU) and is called from host code
__global__ void matrixMul(int* A, int* B, int* C, int N) {
    // Calculate the row and column indices for this thread
    // blockIdx: The block index within the grid
    // blockDim: The dimensions of each block (number of threads per block)
    // threadIdx: The thread index within the block
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    
    // Ensure we only process elements within the matrix bounds
    if (r < N && c < N) {
        int sum = 0;
        // Perform dot product of row r from A and column c from B
        for (int k = 0; k < N; ++k)
            sum += A[r * N + k] * B[k * N + c];
        // Store the result in the output matrix
        C[r * N + c] = sum;
    }
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
    int N;  // Matrix dimension (N×N)
    cout << "Enter matrix size N (NxN): ";
    cin >> N;
    
    // Input validation
    if (N <= 0) {
        cerr << "Error: Matrix size must be positive" << endl;
        return EXIT_FAILURE;
    }
    
    // Calculate total memory size needed for each matrix
    int size = N * N * sizeof(int);
    
    // Allocate host memory for matrices
    int *A = new int[N * N], *B = new int[N * N], *C = new int[N * N];
    
    // Input for matrix A
    cout << "Enter matrix A (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << "A[" << i << "][" << j << "]: ";
            cin >> A[i * N + j];  // Store in row-major order
        }
    }
    
    // Input for matrix B
    cout << "Enter matrix B (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << "B[" << i << "][" << j << "]: ";
            cin >> B[i * N + j];  // Store in row-major order
        }
    }
    
    // Device (GPU) pointers
    int *dA, *dB, *dC;
    
    // Allocate memory on the GPU for each matrix
    checkCudaError(cudaMalloc(&dA, size), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc(&dB, size), "Failed to allocate device memory for B"); 
    checkCudaError(cudaMalloc(&dC, size), "Failed to allocate device memory for C");
    
    // Copy host matrices to device memory
    checkCudaError(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice), 
                  "Failed to copy A from host to device");
    checkCudaError(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice),
                  "Failed to copy B from host to device");
    
    // Define the execution configuration
    // Create a 2D thread block (16×16 threads per block)
    dim3 threads(16, 16);
    // Calculate the grid dimensions to ensure all matrix elements are processed
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    // Record the start event
    checkCudaError(cudaEventRecord(start), "Failed to record start event");
    
    // Launch the CUDA kernel with the defined grid and block dimensions
    matrixMul<<<blocks, threads>>>(dA, dB, dC, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    // Wait for the kernel to complete
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");
    
    // Record the stop event
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    // Wait for the stop event to complete
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    // Calculate the elapsed time between start and stop events
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to get elapsed time");
    
    // Copy the result from device to host
    checkCudaError(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost),
                  "Failed to copy C from device to host");
    
    // Display the resulting matrix
    cout << "\nResult Matrix C:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << C[i * N + j] << " ";
        cout << "\n";
    }
    
    cout << "\nMatrix multiplication completed in " << milliseconds << " ms\n";
    
    // Free device memory
    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC);
    
    // Free CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free host memory
    delete[] A; 
    delete[] B; 
    delete[] C;
    
    return 0;
}

// Sample input and expected flow:
// Enter matrix size N (NxN): 2
// Enter matrix A (2x2):
// A[0][0]: 1
// A[0][1]: 2
// A[1][0]: 3
// A[1][1]: 4
// Enter matrix B (2x2):
// B[0][0]: 5
// B[0][1]: 6
// B[1][0]: 7
// B[1][1]: 8
//
// Result Matrix C:
// 19 22
// 43 50
//
// Matrix multiplication completed in 0.012345 ms