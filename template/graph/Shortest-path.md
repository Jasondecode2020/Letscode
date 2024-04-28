## 1 Dijkstra template

- with node from 1 to n

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = defaultdict(list)
        for u, v, w in times:
            g[u - 1].append((v - 1, w))

        dist = [inf] * n 
        dist[k - 1] = 0
        pq = [(0, k - 1)]
        while pq:
            c, x = heappop(pq)
            if c > dist[x]:
                continue
            for y, cost in g[x]:
                d = dist[x] + cost 
                if d < dist[y]:
                    dist[y] = d 
                    heappush(pq, (d, y))
        res = max(dist)
        return res if res != inf else -1
```

* [743. Network Delay Time](#743-Network-Delay-Time) 1800
* [2642. Design Graph With Shortest Path Calculator](#2642-Design-Graph-With-Shortest-Path-Calculator) 1811
* [1514. Path with Maximum Probability](#1514-Path-with-Maximum-Probability) 1846
* [3123. Find Edges in Shortest Paths](#3123-Find-Edges-in-Shortest-Paths)
* [2473. Minimum Cost to Buy Apples](#2473-Minimum-Cost-to-Buy-Apples)
* [1976. Number of Ways to Arrive at Destination](#1976-Number-of-Ways-to-Arrive-at-Destination)
* [505. The Maze II](#505-The-Maze-II)

1631. 最小体力消耗路径 1948 做法不止一种
1368. 使网格图至少有一条有效路径的最小代价 2069 也可以 0-1 BFS
1786. 从第一个节点出发到最后一个节点的受限路径数 2079
1976. 到达目的地的方案数 2095
2662. 前往目标的最小代价 2154
2045. 到达目的地的第二短时间 2202 也可以 BFS
882. 细分图中的可到达节点 2328
2203. 得到要求路径的最小带权子图 2364
2577. 在网格图中访问一个格子的最少时间 2382
2699. 修改图中的边权 2874
2093. 前往目标城市的最小费用（会员题）
2473. 购买苹果的最低成本（会员题）
2714. 找到最短路径的 K 次跨越（会员题）
2737. 找到最近的标记节点（会员题）

### 743. Network Delay Time

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = defaultdict(list)
        for u, v, c in times:
            g[u - 1].append((v - 1, c))

        pq = [(0, k - 1)]
        dist = [inf] * n 
        dist[k - 1] = 0
        while pq:
            c, x = heappop(pq)
            if c > dist[x]:
                continue
            for y, cost in g[x]:
                d = dist[x] + cost 
                if d < dist[y]:
                    dist[y] = d 
                    heappush(pq, (d, y))
        res = max(dist)
        return res if res != inf else -1
```

### 2642. Design Graph With Shortest Path Calculator

```python
class Graph:

    def __init__(self, n: int, edges: List[List[int]]):
        self.g = defaultdict(list)
        self.n = n 
        for u, v, c in edges:
            self.g[u].append((v, c))

    def addEdge(self, edge: List[int]) -> None:
        u, v, c = edge 
        self.g[u].append((v, c))

    def shortestPath(self, node1: int, node2: int) -> int:
        pq = [(0, node1)]
        dist = [inf] * self.n 
        dist[node1] = 0
        while pq:
            c, x = heappop(pq)
            if x == node2:
                return c 
            if c > dist[x]:
                continue
            for y, cost in self.g[x]:
                d = dist[x] + cost 
                if d < dist[y]:
                    dist[y] = d 
                    heappush(pq, (d, y))
        return -1
```

### 1514. Path with Maximum Probability

```python
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start_node: int, end_node: int) -> float:
        g = defaultdict(list)
        for (u, v), c in zip(edges, succProb):
            g[u].append((v, c))
            g[v].append((u, c))

        pq = [(0, start_node)]
        dist = [inf] * n 
        dist[start_node] = -1
        while pq:
            c, x = heappop(pq)
            if x == end_node:
                return -c
            if c < dist[x]:
                continue
            for y, cost in g[x]:
                d = dist[x] * cost 
                if d < dist[y]:
                    dist[y] = d 
                    heappush(pq, (d, y))
        return 0
```

### 3112. Minimum Time to Visit Disappearing Nodes

```python
class Solution:
    def minimumTime(self, n: int, edges: List[List[int]], disappear: List[int]) -> List[int]:
        g = defaultdict(list)
        for a, b, cost in edges:
            if a != b:
                g[a].append([b, cost])
                g[b].append([a, cost])
        
        dist = [inf] * n
        dist[0], q = 0, [(0, 0)]
        while q:
            cost, node = heapq.heappop(q)
            if cost > dist[node]:
                continue
            for nei, cost in g[node]:
                d = dist[node] + cost
                if d < dist[nei] and d < disappear[nei]:
                    dist[nei] = d
                    heapq.heappush(q, (d, nei))

        ans = [-1] * n
        for i, v in enumerate(dist):
            if v != inf:
                ans[i] = v
        return ans
```

### 3123. Find Edges in Shortest Paths

```python
class Solution:
    def findAnswer(self, n: int, edges: List[List[int]]) -> List[bool]:
        g = defaultdict(list)
        for i, (u, v, w) in enumerate(edges):
            g[u].append((v, w, i))
            g[v].append((u, w, i))
            
        dist = [inf] * n 
        dist[0] = 0
        pq = [(0, 0)]
        while pq:
            c, x = heappop(pq)
            if c > dist[x]:
                continue
            for y, cost, i in g[x]:
                d = dist[x] + cost
                if d < dist[y]:
                    dist[y] = d
                    heappush(pq, (cost, y))
                    
        ans = [False] * len(edges)
        if dist[n - 1] == inf:
            return ans
        visited = set([n - 1])
        def dfs(x):
            for y, cost, i in g[x]:
                if dist[y] + cost == dist[x]:
                    ans[i] = True
                    if y not in visited:
                        visited.add(y)
                        dfs(y)
        dfs(n - 1)
        return ans
```

### 2473. Minimum Cost to Buy Apples

```python
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        g = defaultdict(list)
        for a, b, cost in roads:
            g[a - 1].append([b - 1, cost])
            g[b - 1].append([a - 1, cost])
        
        def dijkstra(start):
            res, dist = inf, [inf] * n
            dist[start], q = 0, [(0, start)]
            while q:
                cost, node = heapq.heappop(q)
                res = min(res, cost * (k + 1) + appleCost[node])
                for nei, cost in g[node]:
                    d = dist[node] + cost
                    if d < dist[nei]:
                        dist[nei] = d
                        heapq.heappush(q, (d, nei))
            return res
        return [dijkstra(i) for i in range(n)]
```

## visited set

```python
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        g = defaultdict(list)
        for u, v, w in roads:
            g[u - 1].append((v - 1, w))
            g[v - 1].append((u - 1, w))
        def dijkstra(i):
            pq, visited, res = [(0, i)], set(), inf
            while pq:
                w, node = heappop(pq)
                if node not in visited:
                    visited.add(node)
                    res = min(res, w + appleCost[node])
                    for nei, weight in g[node]:
                        heappush(pq, (w + weight * (k + 1), nei))
            return res
        return [dijkstra(i) for i in range(n)]
```

### 1976. Number of Ways to Arrive at Destination

```python
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v, t in roads:
            g[u].append((v, t))
            g[v].append((u, t))

        dist = [inf] * n
        dist[0] = 0
        ways = [0] * n
        ways[0] = 1
        pq = [(0, 0)]
        mod = 10 ** 9 + 7
        while pq:
            t, u = heappop(pq)
            for v, c in g[u]:
                time = c + t
                if time < dist[v]:
                    dist[v] = time
                    ways[v] = ways[u]
                    heappush(pq, (time, v))
                elif time == dist[v]:
                    ways[v] += ways[u]
        return ways[-1] % mod
```

### 505. The Maze II

```python
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        def check(r, c, dr, dc):
            dist = 0
            while 0 <= r < R and 0 <= c < C and maze[r][c] != 1:
                r += dr 
                c += dc 
                dist += 1
            return dist - 1, r - dr, c - dc

        pq = [(0, start[0], start[1])]
        R, C = len(maze), len(maze[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        dists = [inf] * R * C 
        dists[start[0] * C + start[1]] = 0
        while pq:
            d, r, c = heappop(pq)
            if r == destination[0] and c == destination[1]:
                return d
            for dr, dc in directions:
                dist, row, col = check(r, c, dr, dc)
                nei = row * C + col
                if dist + d < dists[nei]:
                    dists[nei] = dist + d
                    heappush(pq, (d + dist, row, col))
        return -1
```

## 1 PQ

* [743. Network Delay Time](#743-Network-Delay-Time)
* [2737. Find the Closest Marked Node](#2737. Find the Closest Marked Node)

## 2 Floyd
* [743. Network Delay Time](#743-Network-Delay-Time)
* [1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](#1334-Find-the-City-With-the-Smallest-Number-of-Neighbors-at-a-Threshold-Distance)


## 3 Bellman-Ford

* [787. Cheapest Flights Within K Stops](#787-Cheapest-Flights-Within-K-Stops)
* [743. Network Delay Time](#743-Network-Delay-Time)


```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        pq, d, g = [(0, k)], defaultdict(int), defaultdict(list)
        for u, v, w in times:
            g[u].append((v, w))
        
        while pq:
            time, node = heappop(pq)
            if node not in d:
                d[node] = time
                for nei, t in g[node]:
                    heappush(pq, (time + t, nei))
        return max(d.values()) if len(d) == n else -1
```

### 2737. Find the Closest Marked Node

```python
class Solution:
    def minimumDistance(self, n: int, edges: List[List[int]], s: int, marked: List[int]) -> int:
        pq, d, g = [(0, s)], defaultdict(int), defaultdict(list)
        for u, v, w in edges:
            g[u].append((v, w))
        
        marked = set(marked)
        while pq:
            cost, node = heappop(pq)
            if node not in d:
                d[node] = cost
                for nei, c in g[node]:
                    heappush(pq, (cost + c, nei))
        res = inf
        for k, v in d.items():
            if k in marked:
                res = min(res, v)
        return res if res != inf else -1
```

## 2 Floyd

### 743. Network Delay Time

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[inf] * (n + 1) for r in range(n + 1)]
        for u, v, w in times:
            g[u][v] = w
        for i in range(1, n + 1):
            g[i][i] = 0

        for m in range(1, n + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    g[i][j] = min(g[i][j], g[i][m] + g[m][j])
        res = max(g[k][i] for i in range(1, n + 1))
        return res if res != inf else -1
```

### 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance

```python
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        g = [[inf] * n for r in range(n)]
        for u, v, w in edges:
            g[u][v] = w
            g[v][u] = w
        for i in range(0, n):
            g[i][i] = 0
        for m in range(0, n):
            for i in range(0, n):
                for j in range(0, n):
                    g[i][j] = min(g[i][j], g[i][m] + g[m][j])

        res, ans = 0, inf
        for r in range(n):
            count = 0
            for c in range(n):
                if g[r][c] <= distanceThreshold:
                    count += 1
            if count <= ans:
                ans = count
                res = r
        return res
```

## 3 Bellman-Ford

### 787. Cheapest Flights Within K Stops

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        bf = [inf] * n
        bf[src] = 0
        for i in range(k + 1):
            temp = bf[::]
            for s, d, p in flights:
                temp[d] = min(temp[d], bf[s] + p)
            bf = temp
        return bf[dst] if bf[dst] != inf else -1
```

