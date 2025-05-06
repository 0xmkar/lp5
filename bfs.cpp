#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

int main() {
    int n, e;
    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;
    vector<vector<int>> g(n);
    cout << "Enter edges (u v):\n";
    for (int i = 0, u, v; i < e; ++i) {
        cout << "Edge " << i + 1 << ": ";
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    vector<bool> visited(n, false);
    queue<int> q;
    q.push(0); visited[0] = true;
    cout << "\nParallel BFS traversal starting from node 0:\n";
    while (!q.empty()) {
        int sz = q.size();
        vector<int> next;
        #pragma omp parallel for
        for (int i = 0; i < sz; ++i) {
            int u;
            #pragma omp critical
            { u = q.front(); q.pop(); }

            #pragma omp critical
            cout << u << " ";

            for (int v : g[u]) {
                if (!visited[v]) {
                    #pragma omp critical
                    if (!visited[v]) {
                        visited[v] = true;
                        next.push_back(v);
                    }
                }
            }
        }
        for (int v : next) q.push(v);
    }
    cout << endl;
    return 0;
}

// Enter number of nodes and edges: 4 4
// Enter edges (u v):
// Edge 1: 0 1
// Edge 2: 0 2
// Edge 3: 1 3
// Edge 4: 2 3
