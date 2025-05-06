#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;
void parallelDFS(int start, const vector<vector<int>>& g, vector<bool>& vis) {
    stack<int> s;
    s.push(start);
    while (!s.empty()) {
        int u;
        #pragma omp critical
        { u = s.top(); s.pop(); }

        if (!vis[u]) {
            #pragma omp critical
            { vis[u] = true; cout << u << " "; }

            #pragma omp parallel for
            for (int i = 0; i < g[u].size(); ++i)
                if (!vis[g[u][i]])
                    #pragma omp critical
                    s.push(g[u][i]);
        }
    }
}
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
    vector<bool> vis(n, false);
    cout << "\nParallel DFS traversal starting from node 0:\n";
    parallelDFS(0, g, vis);
    cout << endl;
    return 0;
}

// Enter number of nodes and edges: 6 7
// Enter edges (u v):
// Edge 1: 0 1
// Edge 2: 0 2
// Edge 3: 1 3
// Edge 4: 1 4
// Edge 5: 2 4
// Edge 6: 3 5
// Edge 7: 4 5
