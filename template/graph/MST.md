## Kruscal's Algo

- sort + union find

## Prim's Algo

- graph + pq

## MST question list

* [1135. Connecting Cities With Minimum Cost](#1135-Connecting-Cities-With-Minimum-Cost) 1752
* [1584. Min Cost to Connect All Points](#1584-Min-Cost-to-Connect-All-Points) 1858
* [1168. Optimize Water Distribution in a Village](#1168-Optimize-Water-Distribution-in-a-Village) 2069
* [1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](#1489-Find-Critical-and-Pseudo-Critical-Edges-in-Minimum-Spanning-Tree) 2572

### 1135. Connecting Cities With Minimum Cost

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
    
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        connections.sort(key = lambda x: x[2])
        uf = UF(n)
        res, edges = 0, 0
        for u, v, c in connections:
            if not uf.isConnected(u - 1, v - 1):
                uf.union(u - 1, v - 1)
                edges += 1
                res += c 
        return res if edges == n - 1 else -1
```

```python
class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v, c in connections:
            g[u].append((v, c))
            g[v].append((u, c))

        pq, res, visited = [(0, 1)], 0, set()
        while pq:
            c, x = heappop(pq)
            if x not in visited:
                visited.add(x)
                res += c 
                for y, cost in g[x]:
                    if y not in visited:
                        heappush(pq, (cost, y))
        return res if len(visited) == n else -1
```

### 1584. Min Cost to Connect All Points

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
    
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        edges, n = [], len(points)
        for u in range(n):
            x1, y1 = points[u]
            for v in range(u + 1, n):
                x2, y2=points[v]
                edges.append([u, v, abs(x1 - x2)+abs(y1 - y2)])
        
        edges.sort(key = lambda x: x[2])
        uf, res, edge = UF(n), 0, 0
        for u, v, c in edges:
            if not uf.isConnected(u, v):
                uf.union(u, v)
                edge += 1
                res += c
        return res
```

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        g, n = defaultdict(list), len(points)
        for u in range(n):
            x1, y1 = points[u]
            for v in range(u + 1, n):
                x2, y2 = points[v]
                w = abs(x1 - x2) + abs(y1 - y2)
                g[u].append((w, v))
                g[v].append((w, u))

        pq, res, visited = [(0, 0)], 0, set()
        while len(visited) < n:
            cost, node = heappop(pq)
            if node not in visited:
                res += cost
                visited.add(node)
                for c, nei in g[node]:
                    if nei not in visited:
                        heappush(pq, (c, nei))
        return res
```

### 1168. Optimize Water Distribution in a Village

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
    
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        for i in range(1, n + 1):
            pipes.append((0, i, wells[i - 1]))
        pipes.sort(key = lambda x: x[2])

        uf = UF(n + 1)
        res, edges = 0, 0
        for u, v, c in pipes:
            if not uf.isConnected(u, v):
                uf.union(u, v)
                res += c 
        return res 
```

```python
class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v, c in pipes:
            g[u].append((v, c))
            g[v].append((u, c))
        for i in range(1, n + 1):
            g[0].append((i, wells[i - 1]))
            g[i].append((0, wells[i - 1]))
        
        pq, res, visited = [(0, 0)], 0, set()
        while pq:
            c, x = heappop(pq)
            if x not in visited:
                visited.add(x)
                res += c 
                for y, cost in g[x]:
                    if y not in visited:
                        heappush(pq, (cost, y))
        return res
```

### 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree

```python
```