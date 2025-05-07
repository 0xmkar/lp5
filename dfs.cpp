#include <iostream>  // For input/output operations
#include <vector>    // For storing the adjacency list
#include <stack>     // For DFS stack
#include <omp.h>     // For OpenMP parallel processing
using namespace std;

// Parallel implementation of Depth-First Search
void parallelDFS(int start, const vector<vector<int>>& g, vector<bool>& vis) {
    // Initialize stack with the starting node
    stack<int> s;
    s.push(start);
    
    // Continue until all reachable nodes are processed
    while (!s.empty()) {
        int u;
        // Critical section to safely pop a node from the stack
        // This prevents race conditions when multiple threads access the stack
        #pragma omp critical
        { u = s.top(); s.pop(); }
        
        // Process the node if not already visited
        if (!vis[u]) {
            // Critical section to safely mark node as visited and print
            // This ensures thread-safe access to the shared visited array and output
            #pragma omp critical
            { vis[u] = true; cout << u << " "; }
            
            // Process all neighbors of the current node in parallel
            #pragma omp parallel for
            for (int i = 0; i < g[u].size(); ++i)
                // Check if neighbor is not visited
                if (!vis[g[u][i]])
                    // Critical section to safely push neighbor to stack
                    // This prevents race conditions when updating the stack
                    #pragma omp critical
                    s.push(g[u][i]);
        }
    }
}

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
    vector<bool> vis(n, false);
    
    cout << "\nParallel DFS traversal starting from node 0:\n";
    // Start DFS traversal from node 0
    parallelDFS(0, g, vis);
    
    cout << endl;
    return 0;
}

// Sample input and expected flow:
// Enter number of nodes and edges: 6 7
// Enter edges (u v):
// Edge 1: 0 1
// Edge 2: 0 2
// Edge 3: 1 3
// Edge 4: 1 4
// Edge 5: 2 4
// Edge 6: 3 5
// Edge 7: 4 5
//
// This creates a graph with 6 nodes and 7 edges, and then performs
// a parallel DFS traversal starting from node 0.