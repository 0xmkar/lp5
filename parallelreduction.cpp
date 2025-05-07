#include <iostream>  // For input/output operations
#include <omp.h>     // For OpenMP parallel processing
#include <climits>   // For INT_MAX and INT_MIN constants
using namespace std;

int main() {
    int N;
    cout << "Enter number of elements: ";
    cin >> N;
    
    // Dynamically allocate array for the data
    int *data = new int[N];
    
    cout << "Enter elements:\n";
    for (int i = 0; i < N; ++i) 
        cin >> data[i];
    
    // Initialize variables for reductions
    int sum = 0;            // For calculating sum of elements
    int minVal = INT_MAX;   // Initialize to maximum possible value for finding minimum
    int maxVal = INT_MIN;   // Initialize to minimum possible value for finding maximum
    
    // Parallel for loop with reduction operations
    #pragma omp parallel for reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < N; ++i) {
        // Each thread performs its own reductions on its portion of the data
        sum += data[i];              // Sum reduction
        minVal = min(minVal, data[i]);  // Minimum value reduction
        maxVal = max(maxVal, data[i]);  // Maximum value reduction
        
        // OpenMP combines the partial results from each thread at the end:
        // - sum: adds all partial sums
        // - minVal: takes the minimum of all thread-local minimums
        // - maxVal: takes the maximum of all thread-local maximums
    }
    
    // Print the final results
    cout << "Sum: " << sum << "\nAverage: " << (float)sum / N
         << "\nMin: " << minVal << "\nMax: " << maxVal << endl;
    
    // Free dynamically allocated memory
    delete[] data;
    
    return 0;
}

// Sample input and expected flow:
// Enter number of elements: 6
// Enter elements:
// 12 7 9 21 5 18
// This calculates the sum (72), average (12.0), minimum (5), and maximum (21)
// values for this array in parallel using OpenMP reductions.