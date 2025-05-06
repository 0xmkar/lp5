#include <iostream>
#include <omp.h>
#include <climits>
using namespace std;
int main() {
    int N;
    cout << "Enter number of elements: ";
    cin >> N;
    int *data = new int[N];
    cout << "Enter elements:\n";
    for (int i = 0; i < N; ++i) cin >> data[i];
    int sum = 0, minVal = INT_MAX, maxVal = INT_MIN;
    #pragma omp parallel for reduction(+:sum) reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < N; ++i) {
        sum += data[i];
        minVal = min(minVal, data[i]);
        maxVal = max(maxVal, data[i]);
    }
    cout << "Sum: " << sum << "\nAverage: " << (float)sum / N
         << "\nMin: " << minVal << "\nMax: " << maxVal << endl;
    delete[] data;
    return 0;
}