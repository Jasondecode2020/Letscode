## 1 Simplified Dijkstra PQ

* 743. Network Delay Time
* 2737. Find the Closest Marked Node

### 743. Network Delay Time

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

## 4 Dijkstra template

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