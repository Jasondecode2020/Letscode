## Prim's Algo

- start from any node
- using graph and pq greedy idea

### 1584. Min Cost to Connect All Points

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

        minHeap, res, visited = [(0, 0)], 0, set()
        while len(visited) < n:
            cost, node = heappop(minHeap)
            if node not in visited:
                res += cost
                visited.add((node))
                for c, nei in g[node]:
                    if nei not in visited:
                        heappush(minHeap, (c, nei))
        return res
```

## Kruscal's Algo

- sorted edges then union find nodes

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

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        edges, n = [], len(points)
        for u in range(n):
            x1, y1 = points[u]
            for v in range(u + 1, n):
                x2, y2=points[v]
                edges.append([abs(x1 - x2)+abs(y1 - y2), u, v])
        
        edges.sort()
        uf, res, edge = UF(n), 0, 0
        for c, u, v in edges:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                edge += 1
                res += c
            if edge == n - 1:
                break
        return res
```

### 1135. Connecting Cities With Minimum Cost

- Kruskal's

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

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        connections.sort(key = lambda x: x[2])
        uf, res, edge = UF(n), 0, 0
        for u, v, c in connections:
            if uf.find(u - 1) != uf.find(v - 1):
                uf.union(u - 1, v - 1)
                res += c
                edge += 1
            if edge == n - 1:
                return res
        return -1
```

### 1168. Optimize Water Distribution in a Village

```python
class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        g = defaultdict(list)
        for v in range(n): # add a new well
            g[0].append((wells[v], v + 1))
            g[v + 1].append((wells[v], 0))
        for u, v, w in pipes:
            g[u].append((w, v))
            g[v].append((w, u))

        minHeap, res, visited = [(0, 0)], 0, set()
        while minHeap and len(visited) < n + 1:
            cost, node = heappop(minHeap)
            if node not in visited:
                res += cost
                visited.add((node))
                for c, nei in g[node]:
                    if nei not in visited:
                        heappush(minHeap, (c, nei))
        return res
```