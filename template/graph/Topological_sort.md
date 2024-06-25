## Topological sort(with dp)

* [207. Course Schedule](#207-course-schedule)
* [210. Course Schedule II](#210-Course-Schedule-II)
* [310. Minimum Height Trees](#310-Minimum-Height-Trees)
* [444. Sequence Reconstruction](#444-Sequence-Reconstruction)
* [802. Find Eventual Safe States](#802-Find-Eventual-Safe-States)

* [851. Loud and Rich](#851-loud-and-rich)
* [1059. All Paths from Source Lead to Destination](#1059-All-Paths-from-Source-Lead-to-Destination)
* [1136. Parallel Courses](#1136-Parallel-Courses)
* [1245. Tree Diameter](#1245-tree-diameter)
* [1462. Course Schedule IV](#1462-Course-Schedule-IV)

* [2115. Find All Possible Recipes from Given Supplies](#2115-Find-All-Possible-Recipes-from-Given-Supplies)
* [2192. All Ancestors of a Node in a Directed Acyclic Graph](#2192-all-ancestors-of-a-node-in-a-directed-acyclic-graph)
* [269. Alien Dictionary](#269-Alien-Dictionary)
* [329. Longest Increasing Path in a Matrix](#329-Longest-Increasing-Path-in-a-Matrix)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)

* [2392. Build a Matrix With Conditions](#)
* [1591. Strange Printer II](#1591-strange-printer-ii)
* [2360. Longest Cycle in a Graph](#2360-Longest-Cycle-in-a-Graph)
* [2050. Parallel Courses III](#2050-parallel-courses-iii)
* [1916. Count Ways to Build Rooms in an Ant Colony](#1916)

* [1857. Largest Color Value in a Directed Graph](#1857-largest-color-value-in-a-directed-graph)
* [1557. Minimum Number of Vertices to Reach All Nodes](#1557-Minimum-Number-of-Vertices-to-Reach-All-Nodes)

### 207. Course Schedule

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g, indegree = defaultdict(list), [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indegree[a] += 1

        res, q = 0, deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            node = q.popleft()
            res += 1
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res == numCourses
```

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = defaultdict(list)
        visited = [0] * numCourses
        self.valid = True

        for u, v in prerequisites:
            g[v].append(u)
        
        def dfs(u):
            visited[u] = 1
            for v in g[u]:
                if visited[v] == 0:
                    dfs(v)
                    if not self.valid:
                        return
                elif visited[v] == 1:
                    self.valid = False
                    return
            visited[u] = 2
        
        for i in range(numCourses):
            if self.valid and not visited[i]:
                dfs(i)
        
        return self.valid
```

### 210. Course Schedule II

- compare with 207, only res is different

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        g, indegree = defaultdict(list), numCourses * [0]
        for a, b in prerequisites:
            g[b].append(a)
            indegree[a] += 1

        res, q = [], deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            node = q.popleft()
            res.append(node)
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res if len(res) == numCourses else []
```

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        visited = [0] * numCourses
        self.valid = True
        res = []

        for u, v in prerequisites:
            g[v].append(u)
        
        def dfs(u):
            visited[u] = 1
            for v in g[u]:
                if visited[v] == 0:
                    dfs(v)
                    if not self.valid:
                        return
                elif visited[v] == 1:
                    self.valid = False
                    return
            visited[u] = 2
            res.append(u)
        
        for i in range(numCourses):
            if self.valid and not visited[i]:
                dfs(i)
        
        return res[::-1] if self.valid else []
```

### 310. Minimum Height Trees

- undirected indegree = 1 can pop off

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        g, indegree = defaultdict(list), [0] * n 
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
            indegree[a] += 1
            indegree[b] += 1
        
        q, res = deque([i for i, v in enumerate(indegree) if v == 1]), []
        while q:
            res = list(q)
            for _ in range(len(q)):
                node = q.popleft()
                for nei in g[node]:
                    indegree[nei] -= 1
                    if indegree[nei] == 1:
                        q.append(nei)
        return res if res else list(range(n))
```

### 444. Sequence Reconstruction

```python
class Solution:
    def sequenceReconstruction(self, nums: List[int], sequences: List[List[int]]) -> bool:
        g, indegree = defaultdict(list), [0] * (len(nums) + 1)
        for item in sequences:
            for a, b in pairwise(item):
                g[a].append(b)
                indegree[b] += 1
        
        q, res = deque([i for i, v in enumerate(indegree) if v == 0 and i != 0]), []
        while q:
            if len(q) > 1:
                return False
            node = q.popleft()
            res.append(node)
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res == nums
```


### 802. Find Eventual Safe States

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        g, indegree = defaultdict(list), [0] * len(graph)
        for i, nodes in enumerate(graph):
            for node in nodes:
                g[node].append(i)
                indegree[i] += 1

        q = deque([i for i, d in enumerate(indegree) if d == 0])
        res = []
        while q:
            node = q.popleft()
            res.append(node)
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return sorted(res)
```

### 851. Loud and Rich

```python
class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        g, indegree = defaultdict(list), [0] * len(quiet)
        for a, b in richer:
            g[a].append(b)
            indegree[b] += 1
        
        q, res = deque([i for i, v in enumerate(indegree) if v == 0]), list(range(len(quiet)))
        while q:
            node = q.popleft()
            for nei in g[node]:
                if quiet[res[node]] < quiet[res[nei]]:
                    res[nei] = res[node]
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res
```

### 1059. All Paths from Source Lead to Destination

```python
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        g, indegree = defaultdict(list), [0] * n
        for a, b in edges:
            g[b].append(a)
            indegree[a] += 1
        
        if indegree[destination]:
            return False
        q = deque([destination])
        while q:
            node = q.popleft()
            if node == source:
                return True
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return Falses
```

### 1136. Parallel Courses

```python
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        g, indegree = defaultdict(list), [0] * n
        for prev, nxt in relations:
            g[prev - 1].append(nxt - 1)
            indegree[nxt - 1] += 1

        res, count, q = 0, 0, deque([i for i, d in enumerate(indegree) if d == 0])
        ans = []
        while q:
            res += 1
            for i in range(len(q)):
                node = q.popleft()
                count += 1
                for nei in g[node]:
                    indegree[nei] -= 1
                    if indegree[nei] == 0:
                        q.append(nei)
        return res if res > 0 and count == n else -1
```

### 1245. Tree Diameter

```python
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        g, indegree = defaultdict(list), [0] * (len(edges) + 1)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
            indegree[a] += 1
            indegree[b] += 1
        q, depth, res = deque([i for i, v in enumerate(indegree) if v == 1]), -1, 0
        while q:
            res = len(q)
            for _ in range(res):
                node = q.popleft()
                for nei in g[node]:
                    indegree[nei] -= 1
                    if indegree[nei] == 1:
                        q.append(nei)

```

### 1462. Course Schedule IV

```python
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        g, indegree = defaultdict(list), [0] * numCourses
        for a, b in prerequisites:
            g[a].append(b)
            indegree[b] += 1
        q = deque([i for i, v in enumerate(indegree) if v == 0])
        dp = [[False] * numCourses for r in range(numCourses)]
        while q:
            node = q.popleft()
            for nei in g[node]:
                dp[node][nei] = True
                for i in range(numCourses):
                    dp[i][nei] |= dp[i][node]
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return [dp[a][b] for a, b in queries]
```

### 2115. Find All Possible Recipes from Given Supplies

```python
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        g, indegree = defaultdict(list), defaultdict(int)
        for r, item in zip(recipes, ingredients):
            for i in item:
                g[i].append(r)
                indegree[r] += 1
        
        q, res = deque(supplies), []
        while q:
            node = q.popleft()
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
                    res.append(nei)
        return res
```

### 2192. All Ancestors of a Node in a Directed Acyclic Graph

```python
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        g, indegree = defaultdict(list), [0] * n 
        for a, b in edges:
            g[a].append(b)
            indegree[b] += 1
        
        q, res = deque([i for i, v in enumerate(indegree) if v == 0]), {i: set() for i in range(n)}
        while q:
            node = q.popleft()
            for nei in g[node]:
                res[nei] |= res[node]
                res[nei].add(node)
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return [sorted(list(item)) for item in res.values()]
```

### 269. Alien Dictionary

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        letters = set([c for word in words for c in word])
        g, indegree = defaultdict(list), defaultdict(int)
        for c in letters:
            indegree[c] = 0
        for s, t in pairwise(words):
            flag = False
            for a, b in zip(s, t):
                if a != b:
                    g[a].append(b)
                    indegree[b] += 1
                    flag = True
                    break
            if not flag and len(s) > len(t):
                return ''
        
        q, res = deque([i for i, v in indegree.items() if v == 0]), ''
        while q:
            node = q.popleft()
            res += node
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res if len(res) == len(letters) else ''
```

### 329. Longest Increasing Path in a Matrix

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        @cache
        def dfs(r, c):
            res = 1
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if  0 <= row < R and 0 <= col < C and matrix[row][col] > matrix[r][c]:
                    res = max(res, dfs(row, col) + 1)
            return res

        res = -inf
        for r in range(R):
            for c in range(C):
                res = max(res, dfs(r, c))
        return res
```

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        indegree = [0] * R * C 
        def f(r, c, C):
            return r * C + c

        for r in range(R):
            for c in range(C):
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[f(row, col, C)] += 1

        res, q = 0, deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            for i in range(len(q)):
                node = q.popleft()
                r, c = node // C,  node % C
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[f(row, col, C)] -= 1
                        if indegree[f(row, col, C)] == 0:
                            q.append(f(row, col, C))
            res += 1
        return res
```

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        indegree = [0] * R * C 
        g = defaultdict(list)
        def f(r, c, C):
            return r * C + c

        for r in range(R):
            for c in range(C):
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[f(row, col, C)] += 1
                        g[f(r, c, C)].append(f(row, col, C))

        res, q = 0, deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            for i in range(len(q)):
                node = q.popleft()
                for nei in g[node]:
                    indegree[nei] -= 1
                    if indegree[nei] == 0:
                        q.append(nei)
            res += 1
        return res
```

### 2328. Number of Increasing Paths in a Grid

```python
class Solution:
    def countPaths(self, matrix: List[List[int]]) -> int:
        R, C, res = len(matrix), len(matrix[0]), 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        mod = 10 ** 9 + 7
        @cache
        def dfs(r, c):
            res = 1
            for dr, dc in directions:
                row, col = dr + r, dc + c 
                if 0 <= row < R and 0 <= col < C and matrix[row][col] > matrix[r][c]:
                    res += dfs(row, col)
            return res
        
        for r in range(R):
            for c in range(C):
                res += dfs(r, c)
        return res % mod
```

- topological sort

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        indegree = [0] * R * C 

        for r in range(R):
            for c in range(C):
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[row * C + col] += 1

        res, q = 0, deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            for i in range(len(q)):
                node = q.popleft()
                r, c = node // C,  node % C
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[row * C + col] -= 1
                        if indegree[row * C + col] == 0:
                            q.append(row * C + col)
            res += 1
        return res
```

### 2392. Build a Matrix With Conditions

```python
class Solution:
    def buildMatrix(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        def topologicalSort(conditions):
            g, indegree = defaultdict(list), [0] * (k + 1)
            for a, b in conditions:
                g[a].append(b)
                indegree[b] += 1
            q, res = deque([i for i, v in enumerate(indegree) if v == 0 and i != 0]), []
            while q:
                node = q.popleft()
                res.append(node)
                for nei in g[node]:
                    indegree[nei] -= 1
                    if indegree[nei] == 0:
                        q.append(nei)
            return res if len(res) == k else 0
        row, col = topologicalSort(rowConditions), topologicalSort(colConditions)
        if not row or not col:
            return []
        res = [[0] * k for r in range(k)]
        row_pos = {x: r for r, x in enumerate(row)}
        for c, y in enumerate(col):
            res[row_pos[y]][c] = y
        return res
```

### 1591. Strange Printer II

```python
class Solution:
    def isPrintable(self, targetGrid: List[List[int]]) -> bool:
        R, C = len(targetGrid), len(targetGrid[0])
        color = defaultdict(list)
        for r in range(R):
            for c in range(C):
                color[targetGrid[r][c]].append([r, c])

        g, indegree = defaultdict(list), {c: 0 for c in color}
        for c1 in color:
            x1, x2 = min([item[0] for item in color[c1]]), max([item[0] for item in color[c1]])
            y1, y2 = min([item[1] for item in color[c1]]), max([item[1] for item in color[c1]])
            for c2 in color:
                if c1 != c2:
                    for x, y in color[c2]:
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            g[c1].append(c2)
                            indegree[c2] += 1
                            break

        q, cnt = deque([i for i, v in indegree.items() if v == 0]), 0
        while q:
            node = q.popleft()
            cnt += 1
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return cnt == len(color)
```

### 2360. Longest Cycle in a Graph

```python
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        n, indegree = len(edges), [0] * len(edges)
        for u, v in enumerate(edges):
            if v != -1:
                indegree[v] += 1

        q = deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            node = q.popleft()
            if edges[node] != -1:
                nei = edges[node]
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)

        def cycle(indegree):
            res = -1
            for i, d in enumerate(indegree):
                if d == 1:
                    x, ring = i, 0
                    while True:
                        indegree[x] = -1
                        ring += 1
                        x = edges[x]
                        if x == i:
                            res = max(res, ring)
                            break
            return res
        return cycle(indegree)
```

### 2050. Parallel Courses III

```python
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        g, indegree = defaultdict(list), [0] * n
        for u, v in relations:
            g[u - 1].append(v - 1)
            indegree[v - 1] += 1
        q = deque([i for i, d in enumerate(indegree) if d == 0])
        t = [0] * n
        for node in q:
            t[node] = time[node]

        res = 0
        while q:
            node = q.popleft()
            res = max(res, t[node])
            for nei in g[node]:
                t[nei] = max(t[nei], t[node] + time[nei])
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res
```

### 1916. Count Ways to Build Rooms in an Ant Colony

```python
class Solution:
    def waysToBuildRooms(self, prevRoom: List[int]) -> int:
        mod = 10 ** 9 + 7
        n = len(prevRoom)
        fac, inv = [0] * n, [0] * n 
        fac[0] = inv[0] = 1
        for i in range(1, n):
            fac[i] = fac[i - 1] * i % mod 
            inv[i] = pow(fac[i], mod - 2, mod)
        
        g = defaultdict(list)
        for i in range(1, n):
            g[prevRoom[i]].append(i)

        f, cnt = [0] * n, [0] * n 
        def dfs(u):
            f[u] = 1
            for v in g[u]:
                dfs(v)
                f[u] = f[u] * f[v] * inv[cnt[v]] % mod 
                cnt[u] += cnt[v]
            f[u] = f[u] * fac[cnt[u]] % mod 
            cnt[u] += 1
            return f[0]
        return dfs(0)
```

### 1857. Largest Color Value in a Directed Graph

```python
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        n = len(colors)
        indegree = [0] * n
        for u, v in edges:
            g[u].append(v)
            indegree[v] += 1
        
        q = deque([i for i, v in enumerate(indegree) if v == 0])
        dp = [[0] * 26 for _ in range(n)]
        found = 0
        while q:
            found += 1
            node = q.popleft()
            dp[node][ord(colors[node]) - ord('a')] += 1
            for nei in g[node]:
                for c in range(26):
                    dp[nei][c] = max(dp[nei][c], dp[node][c])
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)

        if found != n:
            return -1
        return max(max(item) for item in dp)
```