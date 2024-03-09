## template 1: bfs + queue

* [207. Course Schedule](#207-course-schedule)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g, indegree = defaultdict(list), [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indegree[a] += 1
        
        res, q = PROBLEM_CONDITION, deque([i for i, v in enumerate(indegree) if v == 0])
        while q:
            node = q.popleft()
            res += 1
            for nei in g[node]:
                indegree[nei] -= 1
                if not indegree[nei]:
                    q.append(nei)
        return PROBLEM_CONDITION
```

## template 2: dfs + stack

* [207. Course Schedule](#207-course-schedule)

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

## Topological sort

* [1557. Minimum Number of Vertices to Reach All Nodes](#1557-Minimum-Number-of-Vertices-to-Reach-All-Nodes)
* [207. Course Schedule](#207-course-schedule)
* [210. Course Schedule II](#210-Course-Schedule-II)
* [1462. Course Schedule IV](#1462-Course-Schedule-IV)
* [2115. Find All Possible Recipes from Given Supplies](#2115-Find-All-Possible-Recipes-from-Given-Supplies)
* [269. Alien Dictionary](#269-Alien-Dictionary)
* [310. Minimum Height Trees](#310-Minimum-Height-Trees)
* [329. Longest Increasing Path in a Matrix](#329-Longest-Increasing-Path-in-a-Matrix)
* [802. Find Eventual Safe States](#802-Find-Eventual-Safe-States)
* [1203](#)
* [1136. Parallel Courses](#1136-Parallel-Courses)
* [1059. All Paths from Source Lead to Destination](#1059-All-Paths-from-Source-Lead-to-Destination)
* [444. Sequence Reconstruction](#444-Sequence-Reconstruction)
* [2360. Longest Cycle in a Graph](#2360-Longest-Cycle-in-a-Graph)
* [2603](#)

## Topological sort + dp

* [2050. Parallel Courses III](#2050. Parallel Courses III)
* [1857](#207-course-schedule)

### 1557. Minimum Number of Vertices to Reach All Nodes

```python
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        indegree = [0] * n 
        for u, v in edges:
            indegree[v] += 1
        return [i for i, v in enumerate(indegree) if v == 0]
```

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

### 1462. Course Schedule IV

```python
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        g, indegree = defaultdict(list), [0] * numCourses
        for a, b in prerequisites:
            g[a].append(b)
            indegree[b] += 1

        searchTable = [[False] * numCourses for r in range(numCourses)]
        q = deque([i for i, v in enumerate(indegree) if v == 0])
        while q:
            node = q.popleft()
            for nei in g[node]:
                searchTable[node][nei] = True
                for i in range(numCourses):
                    searchTable[i][nei] |= searchTable[i][node]
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return [searchTable[i][j] for i, j in queries]
```

### 2115. Find All Possible Recipes from Given Supplies

```python
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        g = defaultdict(list)
        indegree = defaultdict(int)
        for recipe, ingredient in zip(recipes, ingredients):
            for item in ingredient:
                g[item].append(recipe)
                indegree[recipe] += 1
                
        res = []
        q = deque(supplies)
        while q:
            ingredient = q.popleft()
            for recipe in g[ingredient]:
                indegree[recipe] -= 1
                if indegree[recipe] == 0:
                    q.append(recipe)
                    res.append(recipe)
        return res
```

### 269. Alien Dictionary

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        unique_letters = set([c for word in words for c in word])
        g, indegree = defaultdict(list), defaultdict(int)
        for c in unique_letters:
            indegree[c] = 0
        for a, b in pairwise(words):
            for x, y in zip(a, b):
                if x != y:
                    g[x].append(y)
                    indegree[y] += 1
                    break
            else:
                if len(a) > len(b):
                    return ''

        res, q = '', deque([i for i, d in indegree.items() if d == 0])
        while q:
            node = q.popleft()
            res += node
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res if len(res) == len(unique_letters) else ''
```

### 310. Minimum Height Trees

- undirected indegree = 1 can pop off

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        g, indegree = defaultdict(list), [0] * n
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
            indegree[u] += 1
            indegree[v] += 1

        res, q = n, [i for i, d in enumerate(indegree) if d == 1]
        while res > 2:
            res -= len(q)
            ans = []
            for i in range(len(q)):
                node = q.pop()
                for nei in g[node]:
                    indegree[nei] -= 1
                    if indegree[nei] == 1:
                        ans.append(nei)
            q = ans
        return q if q else [0]
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
        for high, low in richer:
            g[high].append(low)
            indegree[low] += 1

        q = deque([i for i, v in enumerate(indegree) if v == 0])
        res = list(range(len(quiet)))
        while q:
            node = q.popleft()
            for nei in g[node]:
                # node has more money than nei
                # check if node is quieter than nei
                if quiet[res[node]] < quiet[res[nei]]:
                    res[nei] = res[node]
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res
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

### 1059. All Paths from Source Lead to Destination

```python
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        g, indegree = defaultdict(list), [0] * n
        for u, v in edges:
            g[v].append(u)
            indegree[u] += 1
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
        return False
```

### 444. Sequence Reconstruction

```python
class Solution:
    def sequenceReconstruction(self, nums: List[int], sequences: List[List[int]]) -> bool:
        g, indegree = defaultdict(list), defaultdict(int)
        unique = set([n for s in sequences for n in s])
        for n in unique:
            indegree[n] = 0
        
        for a in sequences:
            for u, v in pairwise(a):
                g[u].append(v)
                indegree[v] += 1
        res, q = [], [i for i, d in indegree.items() if d == 0]
        while q:
            if len(q) > 1:
                return False
            node = q.pop()
            res.append(node)
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res == nums
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

### 1059. All Paths from Source Lead to Destination

```python
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        g, indegree = defaultdict(list), [0] * n
        for u, v in edges:
            g[v].append(u)
            indegree[u] += 1
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
        return False
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