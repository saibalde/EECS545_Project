import numpy as np

"""
A class for encoding undirected, fully connected graphs, without self-loops
"""
class Graph:
    """
    Graph(nNodes)

    Create a graph with given number of nodes. Initially, the graph is fully
    disconnected.
    """
    def __init__(self, nNodes):
        nEdges = nNodes * (nNodes - 1) // 2

        self.nNodes = nNodes
        self.edgeWt = np.zeros(nEdges)

    """
    Graph.getEdgeWeight(i, j)

    Return the weight of edge between nodes i and j. Nodes are zero-indexed.
    """
    def getEdgeWeight(self, i, j):
        if i < j:
            return self.edgeWt[i * (2 * self.nNodes - i - 1) // 2 + j - i - 1]
        elif i > j:
            return self.edgeWt[j * (2 * self.nNodes - j - 1) // 2 + i - j - 1]
        else:
            raise ValueError('Graph does not have self loops!')

    """
    Graph.setEdgeWeight(i, j, weight)

    Set the weight of edge between nodes i and j. Nodes are zero-indexed
    """
    def setEdgeWeight(self, i, j, weight):
        if i < j:
            self.edgeWt[i * (2 * self.nNodes - i - 1) // 2 + j - i - 1] = weight
        elif i > j:
            self.edgeWt[j * (2 * self.nNodes - j - 1) // 2 + i - j - 1] = weight
        else:
            raise ValueError('Graph does not have self loops!')

    """
    Graph.laplacian(L)

    Compute the graph laplacian and store in in the NumPy array L.
    """
    def laplacian(self, L):
        for i in range(self.nNodes):
            d = 0.0
            for j in range(self.nNodes):
                if i != j:
                    w = self.getEdgeWeight(i, j)
                    d += w
                    L[i, j] = -w
            L[i, i] = d

"""
A class for storing graphs with (partially) labelled graphs. Labels are {+1,-1},
unlabelled nodes are indicated with a label of 0.
"""
class LabelledGraph(Graph):
    """
    LabelledGraph(nNodes)

    Create a graph with given number of nodes. Initially, the graph is fully
    disconnected and all nodes are unlabelled.
    """
    def __init__(self, nNodes):
        Graph.__init__(self, nNodes)
        self.labels = np.zeros(nNodes)

    """
    LabelledGraph.get_label(i)

    Return the label of node i. Nodes are zero-indexed.
    """
    def getLabel(self, i):
        return self.labels[i]

    """
    LabelledGraph.set_label(i, label)

    Set the label of node i. Nodes are zero-indexed.
    """
    def setLabel(self, i, label):
        self.labels[i] = label
