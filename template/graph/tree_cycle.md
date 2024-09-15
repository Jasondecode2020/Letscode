## Tree cycle

* [2360. Longest Cycle in a Graph](#2360-Longest-Cycle-in-a-Graph)
* [2204. Distance to a Cycle in Undirected Graph](#2204-distance-to-a-cycle-in-undirected-graph)

### 2360. Longest Cycle in a Graph

```python
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n = len(edges)
        indegree = [0] * n
        for u, v in enumerate(edges):
            if v >= 0:
                indegree[v] += 1

        q = deque([i for i, v in enumerate(indegree) if v == 0])
        while q:
            node = q.popleft()
            nei = edges[node]
            if nei >= 0:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        
        def cycle(arr):
            res = -1
            for i, v in enumerate(indegree):
                if v == 1:
                    x, cycle = i, 0
                    while True:
                        indegree[x] -= 1
                        cycle += 1
                        x = edges[x]
                        if x == i:
                            res = max(res, cycle)
                            break
            return res
        return cycle(indegree)
```

### 2204. Distance to a Cycle in Undirected Graph

```python
class Solution:
    def distanceToCycle(self, n: int, edges: List[List[int]]) -> List[int]:
        indegree = [0] * n 
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
            indegree[u] += 1
            indegree[v] += 1
        
        q = deque([i for i, v in enumerate(indegree) if v == 1])
        while q:
            node = q.popleft()
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 1:
                    q.append(nei)
        
        res = [0] * n 
        seen = set([i for i, d in enumerate(indegree) if d == 2])
        q = deque([(i, 0) for i, d in enumerate(indegree) if d == 2])
        while q:
            node, depth = q.popleft()
            for nei in g[node]:
                if nei not in seen:
                    seen.add(nei)
                    q.append((nei, depth + 1))
                    res[nei] = depth + 1
        return res
```