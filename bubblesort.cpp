#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
using namespace std;

void bubbleSort(vector<int>& a, bool parallel) {
    int n = a.size();
    for (int i = 0; i < n - 1; ++i)
        #pragma omp parallel for if(parallel)
        for (int j = i % 2; j < n - 1; j += 2)
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
}
void printArray(const vector<int>& arr) {
    for (int x : arr) cout << x << " ";
    cout << "\n";
}
int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> a(n);
    cout << "Enter elements:\n";
    for (int& x : a) cin >> x;

    vector<int> b = a;

    double t1 = omp_get_wtime();
    bubbleSort(a, false);
    cout << "Sequential Time: " << omp_get_wtime() - t1 << " sec\nSorted Array: ";
    printArray(a);

    t1 = omp_get_wtime();
    bubbleSort(b, true);
    cout << "Parallel Time: " << omp_get_wtime() - t1 << " sec\nSorted Array: ";
    printArray(b);

    return 0;
}