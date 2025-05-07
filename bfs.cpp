#include <iostream>    // For input/output operations
#include <vector>      // For storing the adjacency list
#include <queue>       // For BFS queue
#include <omp.h>       // For OpenMP parallel processing
using namespace std;

int main() {
    int n, e;  // n = number of nodes, e = number of edges
    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;
    
    // Create an adjacency list representation of the graph
    // g[u] contains all neighbors of node u
    vector<vector<int>> g(n);
    
    cout << "Enter edges (u v):\n";
    for (int i = 0, u, v; i < e; ++i) {
        cout << "Edge " << i + 1 << ": ";
        cin >> u >> v;
        // Add edges in both directions (undirected graph)
        g[u].push_back(v);
        g[v].push_back(u);
    }
    
    // Initialize visited array to track visited nodes
    vector<bool> visited(n, false);
    
    // Initialize queue for BFS with starting node 0
    queue<int> q;
    q.push(0);
    visited[0] = true;  // Mark starting node as visited
    
    cout << "\nParallel BFS traversal starting from node 0:\n";
    
    // Main BFS loop - continues until queue is empty
    while (!q.empty()) {
        int sz = q.size();  // Number of nodes at current level
        vector<int> next;   // Temporary storage for next level nodes
        
        // Process all nodes at current level in parallel
        #pragma omp parallel for
        for (int i = 0; i < sz; ++i) {
            int u;
            // Critical section to safely remove a node from the queue
            #pragma omp critical
            { u = q.front(); q.pop(); }

            // Critical section to prevent interleaved output
            #pragma omp critical
            cout << u << " ";

            // Explore all neighbors of the current node
            for (int v : g[u]) {
                if (!visited[v]) {
                    // Critical section to safely update shared state
                    #pragma omp critical
                    if (!visited[v]) {  // Double-check in case another thread visited v
                        visited[v] = true;  // Mark neighbor as visited
                        next.push_back(v);  // Store for next level processing
                    }
                }
            }
        }
        
        // Add all nodes for the next level to the queue
        for (int v : next) q.push(v);
    }
    
    cout << endl;
    return 0;
}

// Sample input and expected flow:
// Enter number of nodes and edges: 4 4
// Enter edges (u v):
// Edge 1: 0 1
// Edge 2: 0 2
// Edge 3: 1 3
// Edge 4: 2 3
// This creates a simple graph where node 0 connects to nodes 1 and 2,
// and both nodes 1 and 2 connect to node 3.