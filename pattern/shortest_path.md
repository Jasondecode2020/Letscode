## 1 Dijkstra template

### 1976. Number of Ways to Arrive at Destination

```python
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        g = defaultdict(list)
        mod = 10 ** 9 + 7
        for u, v, t in roads:
            g[u].append((v, t))
            g[v].append((u, t))

        dist, paths = [inf] * n, [0] * n 
        dist[0], paths[0] = 0, 1
        pq = [(0, 0)] # time, node 
        while pq:
            t, node = heappop(pq)
            for nei, c in g[node]:
                time = t + c 
                if time < dist[nei]:
                    dist[nei] = time 
                    heappush(pq, (time, nei))
                    paths[nei] = paths[node]
                elif time == dist[nei]:
                    paths[nei] += paths[node]
        return paths[-1] % mod
```
