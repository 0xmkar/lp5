#include <iostream>  // For input/output operations
#include <vector>    // For storing the array of elements
#include <ctime>     // For time measurement (not used directly in this code)
#include <omp.h>     // For OpenMP parallel processing
using namespace std;

// Bubble sort implementation with option for parallel execution
void bubbleSort(vector<int>& a, bool parallel) {
    int n = a.size();
    for (int i = 0; i < n - 1; ++i)
        // Conditional parallelization based on 'parallel' flag
        #pragma omp parallel for if(parallel)
        // Even-odd sorting pattern: when i is even, we compare elements at even indices
        // When i is odd, we compare elements at odd indices
        for (int j = i % 2; j < n - 1; j += 2)
            // Compare adjacent elements and swap if needed
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
}

// Helper function to print array elements
void printArray(const vector<int>& arr) {
    for (int x : arr) cout << x << " ";
    cout << "\n";
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    // Create and populate the array to be sorted
    vector<int> a(n);
    cout << "Enter elements:\n";
    for (int& x : a) cin >> x;

    // Create a copy of the original array for parallel sorting
    vector<int> b = a;

    // Time and execute sequential bubble sort
    double t1 = omp_get_wtime();  // Get current time
    bubbleSort(a, false);         // Sort without parallelization
    cout << "Sequential Time: " << omp_get_wtime() - t1 << " sec\nSorted Array: ";
    printArray(a);

    // Time and execute parallel bubble sort
    t1 = omp_get_wtime();        // Reset timer
    bubbleSort(b, true);         // Sort with parallelization
    cout << "Parallel Time: " << omp_get_wtime() - t1 << " sec\nSorted Array: ";
    printArray(b);

    return 0;
}

// Sample input and expected flow:
// Enter number of elements: 8
// Enter elements:
// 64 34 25 12 22 11 90 88
// This creates an array of 8 integers and sorts it using both
// sequential and parallel bubble sort algorithms for comparison.