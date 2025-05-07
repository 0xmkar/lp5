#include <iostream>  // For input/output operations
#include <vector>    // For storing the array of elements
#include <ctime>     // For time measurement (not used directly in this code)
#include <omp.h>     // For OpenMP parallel processing
using namespace std;

// Merge function: combines two sorted subarrays into one sorted array
void merge(vector<int>& arr, int l, int m, int r) {
    // Create temporary arrays for the left and right subarrays
    vector<int> L(arr.begin() + l, arr.begin() + m + 1),      // Left subarray: elements from l to m
                R(arr.begin() + m + 1, arr.begin() + r + 1);  // Right subarray: elements from m+1 to r
    
    int i = 0;      // Index for left subarray
    int j = 0;      // Index for right subarray
    int k = l;      // Index for merged array
    
    // Merge the two subarrays by picking the smaller element each time
    while (i < L.size() && j < R.size()) 
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    
    // Copy any remaining elements from the left subarray
    while (i < L.size()) 
        arr[k++] = L[i++];
    
    // Copy any remaining elements from the right subarray
    while (j < R.size()) 
        arr[k++] = R[j++];
}

// Recursive merge sort implementation with parallel option
void mergeSort(vector<int>& arr, int l, int r, bool parallel) {
    // Base case: if the subarray has 0 or 1 elements, it's already sorted
    if (l >= r) return;
    
    // Calculate the middle point to divide the array
    int m = l + (r - l) / 2;
    
    if (parallel) {
        // Parallel execution: create two independent sections that can run in parallel
        #pragma omp parallel sections
        {
            // First section: sort the left half
            #pragma omp section
            mergeSort(arr, l, m, true);
            
            // Second section: sort the right half
            #pragma omp section
            mergeSort(arr, m + 1, r, true);
        }
    } else {
        // Sequential execution: sort left half, then right half
        mergeSort(arr, l, m, false);
        mergeSort(arr, m + 1, r, false);
    }
    
    // Combine the sorted halves
    merge(arr, l, m, r);
}

// Helper function to print array elements
void printArray(const vector<int>& arr) {
    for (int i : arr) cout << i << " ";
    cout << "\n";
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    
    // Create and populate the array to be sorted
    vector<int> arr(n);
    cout << "Enter elements: ";
    for (int& x : arr) cin >> x;
    
    // Create copies of the original array for comparison
    vector<int> seqArr = arr,   // For sequential sorting
                parArr = arr;   // For parallel sorting
    
    // Time and execute sequential merge sort
    double t1 = omp_get_wtime();
    mergeSort(seqArr, 0, n - 1, false);
    cout << "Sequential Time: " << omp_get_wtime() - t1 << " sec\nSorted: ";
    printArray(seqArr);
    
    // Time and execute parallel merge sort
    t1 = omp_get_wtime();
    mergeSort(parArr, 0, n - 1, true);
    cout << "Parallel Time: " << omp_get_wtime() - t1 << " sec\nSorted: ";
    printArray(parArr);
    
    return 0;
}

// Sample input and expected flow:
// Enter number of elements: 10
// Enter elements: 38 27 43 3 9 82 10 76 45 23
// This creates an array of 10 integers and sorts it using both
// sequential and parallel merge sort algorithms for comparison.