## Graph Bipartite

Union Find in Bipartite Graphs:

Bipartite graphs can be identified using Union Find by ensuring that vertices connected by an edge belong to different sets. If any edge connects vertices from the same set, the graph is not bipartite.
This Union Find approach helps in determining if the graph can be divided into two independent groups such that there are no edges within the same group. If such a division is possible, the graph is bipartite; otherwise, it's not.

* 785. Is Graph Bipartite? 1625
* 886. Possible Bipartition 1795


### 785. Is Graph Bipartite?

```python
'''
dfs
'''
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        def dfs(i, c):
            if not self.res:
                return
            color[i] = c 
            for nei in graph[i]:
                if color[nei] != -1:
                    if color[nei] == c:
                        self.res = False 
                        return
                else:
                    dfs(nei, 1 - c)

        n = len(graph)
        color = [-1] * n 
        self.res = True
        for i in range(n):
            if color[i] == -1:
                dfs(i, 0)
                if not self.res:
                    return False
        return True
'''
union find
'''
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p2] = p1

class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        uf = UF(len(graph))
        for x, nodes in enumerate(graph):
            for y in nodes:
                uf.union(nodes[0], y)
                if uf.isConnected(x, y):
                    return False
        return True
```


### 886. Possible Bipartition

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p2] = p1

class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = defaultdict(list)
        for u, v in dislikes:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)

        uf = UF(n)
        for x, nodes in g.items():
            for y in nodes:
                uf.union(nodes[0], y)
                if uf.connected(x, y):
                    return False
        return True
```