## template 1: dfs graph - adjacency matrix

* [547. Number of Provinces](#547-Number-of-Provinces)

```python
def findCircleNum(self, isConnected: List[List[int]]) -> int:
    def dfs(i):
        for j in range(n):
            if isConnected[i][j] == 1 and j not in visited:
                visited.add(j)
                dfs(j)

    res, visited, n = 0, set(), len(isConnected)
    for i in range(n):
        if i not in visited:
            dfs(i)
            res += 1
    return res
```

## template 2: dfs graph - adjacency list

* [1971. Find if Path Exists in Graph](#1971-Find-if-Path-Exists-in-Graph)

```python
def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def dfs(source):
            if source == destination:
                self.res = True
            for nei in g[source]:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)
        
        visited = set([source])
        self.res = False
        dfs(source)
        return self.res
```

## template 3: dfs grid

* [200. Number of Islands](#200-Number-of-Islands)

```python
'''
directions = [[0, 1], [0, -1], [1, 0], [-1, 0]] # 4 directions dfs/bfs
directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)] # 8 directions dfs/bfs
directions = [[1, 0], [0, 1]] # 2 directions for union find
'''
def numIslands(self, grid: List[List[str]]) -> int:
    def dfs(r, c):
        grid[r][c] = '0'
        for dr, dc in directions:
            row, col = r + dr, c + dc 
            if 0 <= row < R and 0 <= col < C and grid[row][col] == '1':
                dfs(row, col)

    R, C, res = len(grid), len(grid[0]), 0
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '1':
                dfs(r, c)
                res += 1
    return res
```

## template4: dfs stack (backtracking)

* [797. All Paths From Source to Target](#797-All-Paths-From-Source-to-Target)

```python
def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res, stack, n = [], [0], len(graph)
        def dfs(i):
            if i == n - 1:
                res.append(stack[::])
                return 
            for nei in graph[i]:
                stack.append(nei)
                dfs(nei)
                stack.pop()
        dfs(0)
        return res
```

## DFS

* [547. Number of Provinces](#547-Number-of-Provinces)
* [1971. Find if Path Exists in Graph](#1971-Find-if-Path-Exists-in-Graph)
* [797. All Paths From Source to Target](#797-All-Paths-From-Source-to-Target)
* [841. Keys and Rooms](#841-Keys-and-Rooms)
* [2316. Count Unreachable Pairs of Nodes in an Undirected Graph](#2316-Count-Unreachable-Pairs-of-Nodes-in-an-Undirected-Graph)
* [1319. Number of Operations to Make Network Connected](#1319-Number-of-Operations-to-Make-Network-Connected)
* [2492. Minimum Score of a Path Between Two Cities](#2492-Minimum-Score-of-a-Path-Between-Two-Cities)
* [2685. Count the Number of Complete Components](#2685-Count-the-Number-of-Complete-Components)
* [2192. All Ancestors of a Node in a Directed Acyclic Graph](#924-Minimize-Malware-Spread)
* [924. Minimize Malware Spread](#924-Minimize-Malware-Spread)
* [802. Find Eventual Safe States](#802-Find-Eventual-Safe-States)
* [261. Graph Valid Tree](#261-Graph-Valid-Tree)
* [1376. Time Needed to Inform All Employees]()

## Grid DFS

* [200. Number of Islands](#200-Number-of-Islands)
* [695. Max Area of Island](#695-max-area-of-island)
* [面试题 16.19. Pond Sizes LCCI](#面试题-1619-pond-sizes-lcci)
* [463. Island Perimeter](#463-island-perimeter)
* [2658. Maximum Number of Fish in a Grid](#2658-maximum-number-of-fish-in-a-grid)
* [1034. Coloring A Border](#1034-coloring-a-border)
* [1020. Number of Enclaves](#1020-number-of-enclaves)
* [1254. Number of Closed Islands](#1254-number-of-closed-islands)
* [130. Surrounded Regions](#130-surrounded-regions)
* [1391. Check if There is a Valid Path in a Grid](#130-surrounded-regions)
* [417. Pacific Atlantic Water Flow](#417-pacific-atlantic-water-flow)
* 529. Minesweeper
* 827. Making A Large Island
* 1905. Count Sub Islands
* [1559. Detect Cycles in 2D Grid](#1559-detect-cycles-in-2d-grid)

### 547. Number of Provinces

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(i):
            for j in range(n):
                if isConnected[i][j] == 1 and j not in visited:
                    visited.add(j)
                    dfs(j)

        res, visited, n = 0, set(), len(isConnected)
        for i in range(n):
            if i not in visited:
                dfs(i)
                res += 1
        return res
```

### 1971. Find if Path Exists in Graph

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def dfs(source):
            if source == destination:
                self.res = True
            for nei in g[source]:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)
        
        visited = set([source])
        self.res = False
        dfs(source)
        return self.res
```

### 797. All Paths From Source to Target

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res, stack, n = [], [0], len(graph)
        def dfs(i):
            if i == n - 1:
                res.append(stack[::])
                return 
            for nei in graph[i]:
                stack.append(nei)
                dfs(nei)
                stack.pop()
        dfs(0)
        return res
```

### 841. Keys and Rooms

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        def dfs(i):
            for nei in rooms[i]:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)
        visited = set([0])
        dfs(0)
        return len(visited) == len(rooms)
```

### 2316. Count Unreachable Pairs of Nodes in an Undirected Graph

```python
class Solution:
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        
        def dfs(i):
            for nei in g[i]:
                if nei not in visited:
                    visited.add(nei)
                    seen.add(nei)
                    dfs(nei)

        res, seen = 0, set()
        for i in range(n):
            if i not in seen:
                seen.add(i)
                visited = set([i])
                dfs(i)
                res += len(visited) * (n - len(seen))
        return res
```

### 1319. Number of Operations to Make Network Connected

```python
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in connections:
            g[u].append(v)
            g[v].append(u)
        
        def dfs(i):
            for nei in g[i]:
                if nei not in visited:
                    visited.add(nei)
                    edges.add((i, nei))
                    edges.add((nei, i))
                    dfs(nei)
                elif (i, nei) not in edges:
                    edges.add((i, nei))
                    edges.add((nei, i))
                    self.res += 1

        self.res = 0
        self.count = 0
        visited = set()
        edges = set()
        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                self.count += 1
        ans = self.count - 1
        return ans if ans <= self.res else -1

# math prove
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in connections:
            g[u].append(v)
            g[v].append(u)
        
        def dfs(i):
            for nei in g[i]:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)

        self.res = 0
        visited = set()
        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                self.res += 1
        return self.res - 1 if len(connections) >= n - 1 else -1
```

### 2492. Minimum Score of a Path Between Two Cities

```python
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v, w in roads:
            g[u].append((v, w))
            g[v].append((u, w))

        def dfs(i):
            for nei, w in g[i]:
                self.res = min(self.res, w)
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)

        visited = set([1])
        self.res = inf
        dfs(1)
        return self.res
```

### 2685. Count the Number of Complete Components

```python
class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        
        def dfs(i):
            for nei in g[i]:
                self.sides += 1
                if nei not in visited:
                    self.n += 1
                    visited.add(nei)
                    dfs(nei)

        self.res = 0
        visited = set()
        for i in range(n):
            if i not in visited:
                visited.add(i)
                self.n = 1
                self.sides = 0
                dfs(i)
                self.res += self.sides == self.n * (self.n - 1)
        return self.res
```

### 2192. All Ancestors of a Node in a Directed Acyclic Graph

```python
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        g = defaultdict(list)
        for u, v in edges:
            g[v].append(u)
        
        d = {i: [] for i in range(n)}
        def dfs(x):
            for y in g[x]:
                if y not in visited:
                    visited.add(y)
                    dfs(y)

        for i in range(n):
            visited = set()
            dfs(i)
            d[i] = sorted(list(visited))
        return list(d.values())
```

### 924. Minimize Malware Spread

```python
class Solution:
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        def dfs(i):
            for j in range(n):
                if graph[i][j] and j not in visited:
                    if j in initial:
                        self.count += 1
                        self.mn = min(self.mn, j)
                    self.total += 1
                    visited.add(j)
                    dfs(j)

        initial = set(initial)
        c = defaultdict(list)
        visited = set()
        n = len(graph)
        for i in range(n):
            if i not in visited:
                self.count = 0
                self.mn = inf
                self.total = 0
                dfs(i)
                if self.count == 1:
                    c[1].append([self.total, self.mn])
                elif self.count > 1:
                    c[2].append([self.total, self.mn])
        if 1 not in c:
            return min(item[1] for item in c[2])
        res, ans = inf, 0
        for total, node in c[1]:
            if total > ans:
                res = node
                ans = total
            if total == ans:
                ans = total
                res = min(res, node)
        return res
```

### 2101. Detonate the Maximum Bombs

```python
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        g = defaultdict(list)
        n = len(bombs)
        for i in range(n):
            x1, y1, r1 = bombs[i]
            for j in range(n):
                x2, y2, r2 = bombs[j]
                if i != j and (x2 - x1) ** 2 + (y2 - y1) ** 2 <= r1 ** 2:
                    g[i].append(j)
        
        def dfs(x):
            for y in g[x]:
                if y not in visited:
                    visited.add(y)
                    dfs(y)
                    
        res = 0
        for i in range(n):
            visited = set([i])
            dfs(i)
            res = max(res, len(visited))
        return res
```

### 802. Find Eventual Safe States

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        n = len(graph)
        color = [0] * n # not visited
        def dfs(i):
            if color[i] > 0:
                return color[i] == 2
            color[i] = 1 # searching node
            for y in graph[i]:
                if not dfs(y):
                    return False
            color[i] = 2 # safe name, already visited
            return True
        return [i for i in range(n) if dfs(i)]
```

### 261. Graph Valid Tree

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        visited = set()
    
        def dfs(i, pa): # cycle checking
            if i in visited: 
                return
            visited.add(i)
            for y in g[i]:
                if y == pa:
                    continue
                if y in visited:
                    return False
                res = dfs(y, i)
                if not res:
                    return False
            return True
        return dfs(0, -1) and len(visited) == n
```

### 1376. Time Needed to Inform All Employees

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        g = defaultdict(list)
        for i, n in enumerate(manager):
            g[n].append(i)
        def dfs(x):
            mx = 0
            for y in g[x]:
                mx = max(mx, dfs(y))
            return mx + informTime[x]
        return dfs(headID)
```


### ####################################################################################### graph grid

### 200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(r, c):
            grid[r][c] = '0'
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == '1':
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        res = 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    dfs(r, c)
                    res += 1
        return res
```

### 695. Max Area of Island

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 0
            self.cnt += 1
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        res = 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    self.cnt = 0
                    dfs(r, c)
                    res = max(res, self.cnt)
        return res
```

### 面试题 16.19. Pond Sizes LCCI

```python
class Solution:
    def pondSizes(self, grid: List[List[int]]) -> List[int]:
        def dfs(r, c):
            grid[r][c] = 1
            self.cnt += 1
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 0:
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        res = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 0:
                    self.cnt = 0
                    dfs(r, c)
                    res.append(self.cnt)
        return sorted(res)
```

### 694. Number of Distinct Islands

```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 0
            for i, (dr, dc) in enumerate(directions):
                row, col = r + dr, c + dc
                self.s += str(i)
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        visited = set()
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    self.s = ''
                    dfs(r, c)
                    visited.add(self.s)
        return len(visited)
```

### 695. Max Area of Island

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 0
            res = 1
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    res += dfs(row, col)
            return res

        R, C, res = len(grid), len(grid[0]), 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    res = max(res, dfs(r, c))
        return res
```

### 463. Island Perimeter

```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        R, C, res = len(grid), len(grid[0]), 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    res += 4
                    for dr, dc in directions:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                            res -= 1
        return res
```

### 2658. Maximum Number of Fish in a Grid

```python
class Solution:
    def findMaxFish(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            self.cnt += grid[r][c]
            grid[r][c] = 0
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] > 0:
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        res = 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] > 0:
                    self.cnt = 0
                    dfs(r, c)
                    res = max(res, self.cnt)
        return res
```

### 1034. Coloring A Border

```python
class Solution:
    def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        def dfs(r, c):
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if (row, col) not in visited:
                    if 0 <= row < R and 0 <= col < C and grid[row][col] == origin:
                        visited.add((row, col))
                        dfs(row, col)
                    else:
                        grid[r][c] = color 

        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        visited = set([(row, col)])
        origin = grid[row][col]
        dfs(row, col)
        return grid
```

### 1020. Number of Enclaves

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        q = deque()
        R, C = len(grid), len(grid[0])
        direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and grid[r][c] == 1:
                    q.append((r, c))
                    grid[r][c] = 0
        
        while q:
            r, c = q.popleft()
            for dr, dc in direction:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    grid[row][col] = 0
                    q.append((row, col))
        return sum(grid[r][c] for r in range(R) for c in range(C))
```

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 0
            self.count += 1
            for dr, dc in direction:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        border = []
        for r in range(R):
            for c in range(C):
                if r == 0 or c == 0 or r == R - 1 or c == C - 1:
                    if grid[r][c] == 1:
                        self.count = 0
                        dfs(r, c)
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    self.count = 0
                    dfs(r, c)
                    res += self.count 
        return res
```

### 1254. Number of Closed Islands

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 1
            if r == 0 or c == 0 or r == R - 1 or c == C - 1:
                self.hasBorder = True
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 0:
                    dfs(row, col)
            return 1 if not self.hasBorder else 0

        res, R, C = 0, len(grid),len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for r in range(1, R - 1):
            for c in range(1, C - 1):
                if grid[r][c] == 0:
                    self.hasBorder = False
                    res += dfs(r, c)
        return res
```

### 130. Surrounded Regions

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        class Solution:
    def solve(self, grid: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def dfs(r, c):
            for dr, dc in direction:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 'O' and (row, col) not in visited:
                    visited.add((row, col))
                    dfs(row, col)

        def dfs2(r, c):
            grid[r][c] = 'X'
            for dr, dc in direction:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 'O':
                    dfs2(row, col)

        R, C = len(grid), len(grid[0])
        direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited = set()
        for r in range(R):
            for c in range(C):
                if r == 0 or c == 0 or r == R - 1 or c == C - 1:
                    if grid[r][c] == 'O':
                        visited.add((r, c))
                        dfs(r, c)
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 'O' and (r, c) not in visited:
                    dfs2(r, c) 
        return grid
```

### 1391. Check if There is a Valid Path in a Grid

```python
class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        def dfs(r, c):
            if r == R - 1 and c == C - 1:
                self.res = True
                return
            if (r, c) in visited:
                return
            visited.add((r, c))
            e = grid[r][c]
            if e in [1, 4, 6] and c + 1 < C and grid[r][c + 1] in [1, 3, 5]:
                dfs(r, c + 1)
            if e in [1, 3, 5] and c - 1 >= 0 and grid[r][c - 1] in [1, 4, 6]:
                dfs(r, c - 1)
            if e in [2, 5, 6] and r - 1 >= 0 and grid[r - 1][c] in [2, 3, 4]:
                dfs(r - 1, c)
            if e in [2, 3, 4] and r + 1 < R and grid[r + 1][c] in [2, 5, 6]:
                dfs(r + 1, c)

        R, C = len(grid), len(grid[0])
        visited = set()
        self.res = False
        dfs(0, 0)
        return self.res
```

### 417. Pacific Atlantic Water Flow

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        R, C = len(heights), len(heights[0])
        P, A = set(), set()
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        
        def dfs(r, c, visited, prevHeight):
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and heights[row][col] >= prevHeight:
                    visited.add((row, col))
                    dfs(row, col, visited, heights[row][col])
            
        for c in range(C):
            P.add((0, c))
            dfs(0, c, P, heights[0][c])
            A.add((R - 1, c))
            dfs(R - 1, c, A, heights[R - 1][c])
        for r in range(R):
            P.add((r, 0))
            dfs(r, 0, P, heights[r][0])
            A.add((r, C - 1))
            dfs(r, C - 1, A, heights[r][C - 1])
        
        return [(r, c) for r in range(R) for c in range(C) if (r, c) in P and (r, c) in A]
```

### 1559. Detect Cycles in 2D Grid

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
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R, C = len(grid), len(grid[0])
        directions = [[1, 0], [0, 1]]
        uf = UF(R * C)
        res = 0
        for r in range(R):
            for c in range(C):
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C and grid[row][col] == grid[r][c]:
                        if not uf.isConnected(row * C + col, r * C + c):
                            uf.union(row * C + col, r * C + c)
                        else:
                            return True
        return False
```

### 529. Minesweeper

```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        if board[click[0]][click[1]] == 'M':
            board[click[0]][click[1]] = 'X'
            return board
        R, C = len(board), len(board[0])
        def check(i, j):
            cnt = 0
            for x, y in directions:
                x, y = x + i, y + j
                if 0 <= x < R and 0 <= y < C and board[x][y] == 'M':
                    cnt += 1
            return cnt    
        def dfs(i, j):
            cnt = check(i, j)
            if not cnt:
                board[i][j] = 'B'
                for x, y in directions:
                    x, y = x + i, y + j
                    if  0 <= x < R and 0 <= y < C and board[x][y] == 'E': 
                        dfs(x, y)
            else: 
                board[i][j] = str(cnt)
        dfs(click[0],click[1])
        return board
```

### 827. Making A Large Island

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
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        uf = UF(R * C)
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    for dr, dc in [[1, 0], [0, 1]]:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and grid[row][col] and not uf.connected(row * C + col, r * C + c):
                            uf.union(row * C + col, r * C + c)
        
        res = max(uf.rank)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 0:
                    visited = set()
                    ans = 1
                    for a, b in directions:
                        x, y = i + a, j + b
                        if 0 <= x < R and 0 <= y < C and grid[x][y]:
                            root = uf.find(x * C + y)
                            if root not in visited:
                                visited.add(root)
                                ans += uf.rank[root]
                    res = max(res, ans)
        return res
```

### 1905. Count Sub Islands

```python
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        def dfs(r, c):
            if r < 0 or c < 0 or r == R or c == C or grid2[r][c] == 0:
                return
            grid2[r][c] = 0
            if grid1[r][c] == 0:
                self.isSub = False
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        R, C = len(grid2), len(grid2[0])
        res = 0
        for r in range(R):
            for c in range(C):
                self.isSub = True
                if grid2[r][c] == 1:
                    dfs(r, c)
                    if self.isSub: res += 1
                    
        return res
```

### 1631, 778, 1036, 1263, 2258, 2577, 1728 hard

- other graph

### 2246. Longest Path With Different Adjacent Characters

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        g = defaultdict(list)
        n = len(parent)
        for i in range(1, n):
            g[parent[i]].append(i)

        self.res = 0
        def dfs(x):
            x_len = 0
            for y in g[x]:
                y_len = dfs(y) + 1
                if s[y] != s[x]:
                    # longest and second longest
                    self.res = max(self.res, x_len + y_len)
                    x_len = max(x_len, y_len)
            return x_len
        dfs(0)
        return self.res + 1
```

### 1245. Tree Diameter

```python
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        self.res = 0
        def dfs(x, parent): # dfs(0), x = 0, parent = -1, dfs(1, 0)
            first = 0
            for y in g[x]:
                if y == parent:
                    continue
                second = dfs(y, x) + 1
                self.res = max(self.res, first + second)
                first = max(first, second)
            return first
        dfs(0, -1)
        return self.res
```

### 419. Battleships in a Board

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        R, C = len(board), len(board[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        res = 0
        def dfs(r, c):
            board[r][c] = '.'
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and board[row][col] == 'X':
                    dfs(row, col)
    
        for r in range(R):
            for c in range(C):
                if board[r][c] == 'X':
                    dfs(r, c)
                    res += 1
        return res
```

### 1340. Jump Game V

```python
class Solution:
    def maxJumps(self, arr: List[int], d: int) -> int:
        g = defaultdict(list)
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[j] < arr[i] and abs(j - i) <= d:
                    g[i].append(j)
                else:
                    break
            for j in range(i - 1, -1, -1):
                if arr[j] < arr[i] and abs(j - i) <= d:
                    g[i].append(j)
                else:
                    break
        @cache
        def dfs(i):
            res = 1
            for j in g[i]:
                res = max(res, 1 + dfs(j))
            return res 
            
        res = 0
        for i in range(n):
            res = max(res, dfs(i))
        return res
```

### 3249. Count the Number of Good Nodes

```python
class Solution:
    def countGoodNodes(self, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        def dfs(x, fa):
            size, sz0, ok = 1, 0, True
            for y in g[x]:
                if y == fa:
                    continue
                sz = dfs(y, x)
                if sz0  == 0:
                    sz0 = sz 
                elif sz != sz0:
                    ok = False 
                size += sz 
            self.res += ok
            return size 
        self.res = 0
        dfs(0, -1)
        return self.res
```