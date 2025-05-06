#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
using namespace std;

void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> L(arr.begin() + l, arr.begin() + m + 1), R(arr.begin() + m + 1, arr.begin() + r + 1);
    int i = 0, j = 0, k = l;
    while (i < L.size() && j < R.size()) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < L.size()) arr[k++] = L[i++];
    while (j < R.size()) arr[k++] = R[j++];
}
void mergeSort(vector<int>& arr, int l, int r, bool parallel) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    if (parallel) {
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m, true);
            #pragma omp section
            mergeSort(arr, m + 1, r, true);
        }
    } else {
        mergeSort(arr, l, m, false);
        mergeSort(arr, m + 1, r, false);
    }
    merge(arr, l, m, r);
}
void printArray(const vector<int>& arr) {
    for (int i : arr) cout << i << " ";
    cout << "\n";
}
int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter elements: ";
    for (int& x : arr) cin >> x;
    vector<int> seqArr = arr, parArr = arr;
    double t1 = omp_get_wtime();
    mergeSort(seqArr, 0, n - 1, false);
    cout << "Sequential Time: " << omp_get_wtime() - t1 << " sec\nSorted: ";
    printArray(seqArr);
    t1 = omp_get_wtime();
    mergeSort(parArr, 0, n - 1, true);
    cout << "Parallel Time: " << omp_get_wtime() - t1 << " sec\nSorted: ";
    printArray(parArr);
    return 0;
}