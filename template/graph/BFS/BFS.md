## template 1

```python
from collections import deque

def fn(g): # graph
    q = deque([START_NODE]) # q: deque
    visited = {START_NODE}
    res = 0

    while q:
        node = q.popleft()
        # some code
        for neighbor in g[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return res
```

## template 2: 994. Rotting Oranges

```python
def fn(q):
    while q:
        r, c, t = q.popleft()
        res = max(res, t)
        for dr, dc in directions:
            row, col = r + dr, c + dc 
            if 0 <= row < R and 0 <= col < C and (row, col) and grid[row][col] == 1:
                q.append((row, col, t + 1))
                grid[row][col] = 2
```

* [1306. Jump Game III](#1306-Jump-Game-III)
* [317. Shortest Distance from All Buildings](#317-Shortest-Distance-from-All-Buildings)
* [863. All Nodes Distance K in Binary Tree](#863-All-Nodes-Distance-K-in-Binary-Tree)
* [1345. Jump Game IV](#1345-Jump-Game-IV)
* [1730. Shortest Path to Get Food](#1730-Shortest-Path-to-Get-Food)
* [1466. Reorder Routes to Make All Paths Lead to the City Zero](#1466-Reorder-Routes-to-Make-All-Paths-Lead-to-the-City-Zero)
* [433. Minimum Genetic Mutation](#433-Minimum-Genetic-Mutation)
* [1311. Get Watched Videos by Your Friends](#1311-Get-Watched-Videos-by-Your-Friends)
* [1129. Shortest Path with Alternating Colors](#1129-Shortest-Path-with-Alternating-Colors)
* [2608. Shortest Cycle in a Graph](#2608-Shortest-Cycle-in-a-Graph)
* [1298. Maximum Candies You Can Get from Boxes](#1298-Maximum-Candies-You-Can-Get-from-Boxes)
* [631. Design Excel Sum Formula](#631-design-excel-sum-formula)
* [1377. Frog Position After T Seconds](#1377-frog-position-after-t-seconds)

### 542. 01 Matrix

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        visited = set()
        q = deque()
        directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        for r in range(R):
            for c in range(C):
                if mat[r][c] == 0:
                    visited.add((r, c))
                    q.append((r, c, 0))

        while q:
            r, c, d = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and mat[row][col] == 1 and (row, col) not in visited:
                    mat[row][col] = d + 1
                    visited.add((row, col))
                    q.append((row, col, d + 1))
        return mat
```

### 994. Rotting Oranges

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        visited = set()
        q = deque()
        directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 2:
                    visited.add((r, c))
                    q.append((r, c, 0))

        res = 0
        while q:
            r, c, d = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1 and (row, col) not in visited:
                    res = max(res, d + 1)
                    visited.add((row, col))
                    q.append((row, col, d + 1))
        return res if len(visited) == sum(grid[r][c] > 0 for r in range(R) for c in range(C)) else -1
```

### 2684. Maximum Number of Moves in a Grid

```python
class Solution:
    def maxMoves(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        visited = set([(r, 0) for r in range(R)])
        q = deque([(r, 0, 0) for r in range(R)])

        res = 0
        while q:
            r, c, d = q.popleft()
            res = max(res, d)
            for row, col in [(r - 1, c + 1), (r, c + 1), (r + 1, c + 1)]:
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and grid[r][c] < grid[row][col]:
                    visited.add((row, col))
                    q.append((row, col, d + 1))
        return res 
```

### 1926. Nearest Exit from Entrance in Maze

```python
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        R, C = len(maze), len(maze[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        visited = set()
        q = deque()
        for r in range(R):
            for c in range(C):
                if [r, c] != entrance and (r == 0 or r == R - 1 or c == 0 or c == C - 1) and maze[r][c] == '.':
                    visited.add((r, c))
                    q.append((r, c, 0))

        res = -1
        while q:
            r, c, d = q.popleft()
            if [r, c] == entrance:
                res = d
                break 
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and maze[row][col] == '.':
                    visited.add((row, col))
                    q.append((row, col, d + 1))
        return res
```

### 1162. As Far from Land as Possible

```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        q = deque()
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        R, C = len(grid), len(grid[0])
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    q.append((r, c, 0))
        res = 0
        while q:
            r, c, d = q.popleft()
            res = d
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 0:
                    grid[row][col] = 1
                    q.append((row, col, d + 1))
        return res if res else -1
```

### 934. Shortest Bridge

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        def dfs(r, c):
            grid[r][c] = 2
            heappush(pq, (0, r, c))
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    dfs(row, col)

        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    pq = []
                    dfs(r, c)
                    while pq:
                        d, row, col = heappop(pq)
                        for dr, dc in directions:
                            new_row, new_col = row + dr, col + dc 
                            if 0 <= new_row < R and 0 <= new_col < C:
                                if grid[new_row][new_col] == 1:
                                    return d 
                                if grid[new_row][new_col] == 0:
                                    grid[new_row][new_col] = 2
                                    heappush(pq, (d + 1, new_row, new_col))
```

### 1293. Shortest Path in a Grid with Obstacles Elimination

```python
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        if grid[0][0] == 1 and k == 0:
            return -1
        q = deque([(0, 0, 0 if grid[0][0] == 0 else 1, 0)])
        visited = set([(0, 0, 0 if grid[0][0] == 0 else 1)])
        directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        R, C = len(grid), len(grid[0])
        while q:
            r, c, obstacles, steps = q.popleft()
            if r == R - 1 and c == C - 1:
                return steps
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col, obstacles) not in visited:
                    visited.add((row, col, obstacles))
                    if grid[row][col] == 0:
                        q.append((row, col, obstacles, steps + 1))
                    else:
                        if obstacles + 1 <= k:
                            q.append((row, col, obstacles + 1, steps + 1))
        return -1
```


### 317. Shortest Distance from All Buildings

```python
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        q, cnt = deque(), 0
        distance = [[0] * C for r in range(R)]
        counts = [[0] * C for r in range(R)]
        res = inf
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 1:
                    cnt += 1
                    q.append((i, j, 0))
                    visited = [[False] * C for r in range(R)]
                    while q:
                        r, c, dist = q.popleft()
                        for dr, dc in directions:
                            row, col = r + dr, c + dc 
                            if 0 <= row < R and 0 <= col < C and grid[row][col]==0 and not visited[row][col]:
                                counts[row][col] += 1
                                distance[row][col] += dist + 1
                                q.append((row, col, dist+1))
                                visited[row][col] = True
        for r in range(R):
            for c in range(C):
                if counts[r][c] == cnt and distance[r][c] < res:
                    res = distance[r][c]
        return res if res != inf else -1
```

### 1306. Jump Game III

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        q = deque([start])
        visited = set([start])
        while q:
            node = q.popleft()
            if arr[node] == 0:
                return True
            for nei in [node + arr[node], node - arr[node]]:
                if 0 <= nei < len(arr) and nei not in visited:
                    visited.add(nei)
                    q.append(nei)
        return False
```

### 317. Shortest Distance from All Buildings

- brute force, O(n^4) can't pass

```python
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        target = sum(row.count(1) for row in grid)
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        def bfs(r, c):
            q = deque([(r, c, 0)])
            visited = set([(r, c)])
            res, count = 0, 0
            while q:
                r, c, d = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and (row, col) not in visited:
                        if grid[row][col] == 1:
                            res += d + 1
                            visited.add((row, col))
                            count += 1
                        if grid[row][col] == 0:
                            visited.add((row, col))
                            q.append((row, col, d + 1))
            if count == target:
                return res 
            else:
                for r, c in visited:
                    if grid[r][c] == 0:
                        grid[r][c] = 2
            return inf 
        res = inf
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 0:
                    res = min(res, bfs(r, c))
        return res if res != inf else -1
```

- use visited matrix can pass

```python
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        target = sum(row.count(1) for row in grid)
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        def bfs(r, c):
            q = deque([(r, c, 0)])
            visited = [[False] * C for r in range(R)]
            res, count = 0, 0
            while q:
                r, c, d = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and not visited[row][col]:
                        if grid[row][col] == 1:
                            res += d + 1
                            visited[row][col] = True
                            count += 1
                        if grid[row][col] == 0:
                            visited[row][col] = True
                            q.append((row, col, d + 1))
            if count == target:
                return res 
            else:
                for x in range(R):
                    for y in range(C):
                        if grid[x][y] == 0 and visited[x][y]:
                            grid[x][y] = 2
            return inf 
        res = inf
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 0:
                    res = min(res, bfs(r, c))
        return res if res != inf else -1
```

```python
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        q, cnt = deque(), 0
        distance = [[0] * C for r in range(R)]
        counts = [[0] * C for r in range(R)]
        res = inf
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 1:
                    cnt += 1
                    q.append((i, j, 0))
                    visited = [[False] * C for r in range(R)]
                    while q:
                        r, c, dist = q.popleft()
                        for dr, dc in directions:
                            row, col = r + dr, c + dc 
                            if 0 <= row < R and 0 <= col < C and grid[row][col]==0 and not visited[row][col]:
                                counts[row][col] += 1
                                distance[row][col] += dist + 1
                                q.append((row, col, dist+1))
                                visited[row][col] = True
        for r in range(R):
            for c in range(C):
                if counts[r][c] == cnt and distance[r][c] < res:
                    res = distance[r][c]
        return res if res != inf else -1
```

### 863. All Nodes Distance K in Binary Tree

```python
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        g = defaultdict(list)
        q = deque([root])
        while q:
            node = q.popleft()
            if node.left:
                q.append(node.left)
                g[node.val].append(node.left.val)
                g[node.left.val].append(node.val)
            if node.right:
                q.append(node.right)
                g[node.val].append(node.right.val)
                g[node.right.val].append(node.val)
        
        q = deque([(target.val, 0)])
        visited = set([target.val])
        res = []
        while q:
            node, d = q.popleft()
            if d == k:
                res.append(node)
                continue
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, d + 1))
        return res
```

### 1345. Jump Game IV

```python
class Solution:
    def minJumps(self, arr: List[int]) -> int:
        d = defaultdict(list)
        n = len(arr)
        for i, v in enumerate(arr):
            d[v].append(i)

        visited = [False] * n 
        q = deque([(0, 0)]) # idx, steps
        visited[0] = True
        while q:
            idx, steps = q.popleft()
            if idx == n - 1:
                return steps 
            if idx - 1 >= 0 and not visited[idx - 1]:
                visited[idx - 1] = True
                q.append((idx - 1, steps + 1))
            if idx + 1 < n and not visited[idx + 1]:
                visited[idx + 1] = True
                q.append((idx + 1, steps + 1))
            for i in d[arr[idx]]:
                if not visited[i]:
                    visited[i] = True
                    q.append((i, steps + 1))
            d.pop(arr[idx])
```

### 1730. Shortest Path to Get Food

```python
R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '*':
                    q = deque([(r, c, 0)])
                    visited = set()
                    visited.add((r, c))
                    while q:
                        r, c, steps = q.popleft()
                        if grid[r][c] == '#':
                            return steps 
                        for dr, dc in directions:
                            row, col = r + dr, c + dc 
                            if 0 <= row < R and 0 <= col < C and grid[row][col] != 'X' and (row, col) not in visited:
                                q.append((row, col, steps + 1))
                                visited.add((row, col))
        return -1
```

### 1466. Reorder Routes to Make All Paths Lead to the City Zero

```python
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        d = defaultdict(bool)
        g = defaultdict(list)
        for u, v in connections:
            d[(u, v)] = True
            d[(v, u)] = False 
            g[u].append(v)
            g[v].append(u)
        
        q = deque([0])
        visited = set([0])
        res = 0
        while q:
            node = q.popleft()
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append(nei)
                    if d[(node, nei)]:
                        res += 1
        return res
```

### 433. Minimum Genetic Mutation

```python
bank = set(bank)
        q = deque([(startGene, 0)])
        s = set([startGene])
        g = {
            'A': ['C', 'G', 'T'], 
            'C': ['A', 'G', 'T'], 
            'G': ['C', 'A', 'T'], 
            'T': ['C', 'G', 'A']
        }
        while q:
            gene, steps = q.popleft()
            if gene == endGene:
                return steps
            for i in range(len(gene)):
                for nei in g[gene[i]]:
                    nei_gene = gene[:i] + nei + gene[i+1:]
                    if nei_gene not in s and nei_gene in bank:
                        s.add(nei_gene)
                        q.append((nei_gene, steps + 1))
        return -1
```

### 1311. Get Watched Videos by Your Friends

```python
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        g = defaultdict(list)
        for i, a in enumerate(friends):
            for j in a:
                g[i].append(j)
                g[j].append(i)

        q = deque([(id, 0)])
        visited = set([id])
        c = Counter()
        while q:
            idx, depth = q.popleft()
            if depth == level:
                for v in watchedVideos[idx]:
                    c[v] += 1
            for nei in g[idx]:
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, depth + 1))
        res = []
        for u, v in c.items():
            res.append((v, u))
        res.sort()
        return [item[1] for item in res]

```

### 1129. Shortest Path with Alternating Colors

```python
class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for u, v in redEdges:
            g[u].append((v, 0))
        for u, v in blueEdges:
            g[u].append((v, 1))

        res = [-1] * n
        visited = set([(0, -1)])
        q = deque([(0, -1, 0)])
        while q:
            node, color, depth = q.popleft()
            if res[node] != -1:
                res[node] = min(res[node], depth)
            else:
                res[node] = depth 
            for nei, c in g[node]:
                if c != color and (nei, c) not in visited:
                    visited.add((nei, c))
                    q.append((nei, c, depth + 1))
        return res
```

### 2608. Shortest Cycle in a Graph

```python
class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        
        def bfs(start):
            res = inf
            dist = [-1] * n 
            dist[start] = 0
            q = deque([(start, -1)])
            while q:
                x, fa = q.popleft()
                for y in g[x]:
                    if dist[y] < 0:
                        dist[y] = dist[x] + 1
                        q.append((y, x))
                    elif y != fa:
                        res = min(res, dist[x] + dist[y] + 1)
            return res 
        res = min(bfs(i) for i in range(n))
        return res if res != inf else -1
```

### 1298. Maximum Candies You Can Get from Boxes

```python
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        has_box, has_key, visited_box = set(), set(), set()
        q = deque()
        for box in initialBoxes:
            has_box.add(box)
            if status[box]:
                visited_box.add(box)
                q.append(box)
        res = 0
        while q:
            big_box = q.popleft()
            res += candies[big_box]
            for key in keys[big_box]:
                has_key.add(key)
                if key in has_box and key not in visited_box:
                    visited_box.add(key)
                    q.append(key)
            for box in containedBoxes[big_box]:
                has_box.add(box)
                if not box in visited_box:
                    if status[box] or box in has_key:
                        visited_box.add(box)
                        q.append(box)
        return res
```

### 631. Design Excel Sum Formula

```python
class Excel:
    def __init__(self, height: int, width: str):
        self.g = [[0] * (ord(width) - 64) for r in range(height)]
        self.sum_state = deepcopy(self.g)

    def set(self, row: int, column: str, val: int) -> None:
        self.g[row - 1][ord(column) - 65] = val
        # 0 means single val, otherwise: array
        self.sum_state[row - 1][ord(column) - 65] = 0 

    def get(self, row: int, column: str) -> int:
        if self.sum_state[row - 1][ord(column) - 65] == 0:
            return self.g[row - 1][ord(column) - 65]
        q, res = deque([(row - 1, ord(column) - 65)]), 0
        while q:
            r, c = q.popleft()
            if self.sum_state[r][c] == 0:
                res += self.g[r][c]
            else:
                for s in self.sum_state[r][c]:
                    if len(s) <= 3:
                        row, col = int(s[1:]) - 1, ord(s[0]) - 65
                        q.append((row, col))
                    else:
                        a, b = s.split(':')
                        row1, col1 = int(a[1:]) - 1, ord(a[0]) - 65
                        row2, col2 = int(b[1:]) - 1, ord(b[0]) - 65
                        for i in range(row1, row2 + 1):
                            for j in range(col1, col2 + 1):
                                q.append((i, j))
        return res

    def sum(self, row: int, column: str, numbers: List[str]) -> int:
        self.sum_state[row - 1][ord(column) - 65] = numbers
        return self.get(row, column)
```

### 1377. Frog Position After T Seconds

```python
class Solution:
    def frogPosition(self, n: int, edges: List[List[int]], t: int, target: int) -> float:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        q = deque([(1, 1, 0)])
        visited = set([1])
        res = 1
        while q:
            node, prob, time = q.popleft()
            if node == target and (time < t and all(n in visited for n in g[node]) or time == t):
                return prob
            for nei in g[node]:
                if nei not in visited:
                    visited.add(nei)
                    if node == 1:
                        q.append((nei, prob * (1 / len(g[node])), time + 1))
                    else:
                        q.append((nei, prob * (1 / (len(g[node]) - 1)), time + 1))
        return 0
```

### 2608. Shortest Cycle in a Graph

```python
class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
            
        def bfs(u, v):
            q = deque([(u, 0)])
            visited = set([u])
            while q:
                node, dist = q.popleft()
                if node == v:
                    return dist + 1
                for nei in g[node]:
                    if nei not in visited:
                        visited.add(nei)
                        q.append((nei, dist + 1))
            return inf 

        res = inf 
        for u, v in edges:
            g[u].remove(v)
            g[v].remove(u)
            res = min(res, bfs(u, v))
            g[u].append(v)
            g[v].append(u)
        return res if res != inf else -1
```

### 1215. Stepping Numbers

```python
class Solution:
    def countSteppingNumbers(self, low: int, high: int) -> List[int]:
        res = []
        if low == 0:
            res.append(0)
        
        q = deque([i for i in range(1, 10)])
        while q:
            n = q.popleft()
            if n > high:
                return res 
            if n >= low:
                res.append(n)
            lastDigit = n % 10
            if lastDigit > 0:
                q.append(n * 10 + lastDigit - 1)
            if lastDigit < 9:
                q.append(n * 10 + lastDigit + 1)
```

### 127. Word Ladder

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if endWord not in wordSet:
            return 0
        l, s1, s2 = len(beginWord), {beginWord}, {endWord}
        wordSet.remove(endWord)

        step = 1
        while s1:
            step += 1
            s = set()
            for w in s1:
                words = [w[:i] + c + w[i + 1:] for c in ascii_lowercase for i in range(l)]
                for word in words:
                    if word in s2:
                        return step
                    if word in wordSet:
                        s.add(word)
            for word in s:
                wordSet.remove(word)
            s1 = s 
        return 0
```

### 126. Word Ladder II

```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        wordSet = set(wordList)
        ans = []
        if endWord not in wordSet:
            return ans
        l, s1, s2 = len(beginWord), {beginWord}, {endWord}
        flag, d = False, defaultdict(list)
        while s1:
            s = set()
            for w in s1:
                words = [w[:i] + c + w[i + 1:] for c in ascii_lowercase for i in range(l)]
                for word in words:
                    if word in s2:
                        flag = True
                    if word in wordSet:
                        s.add(word)
                        d[word].append(w)
            for word in s:
                wordSet.remove(word)
            s1 = s 
            if flag:
                break 
        def backtrack(cur, res):
            if cur == beginWord:
                ans.append(res[::-1])
                return 
            for word in d[cur]:
                backtrack(word, res + [word])
            return ans 
        return backtrack(endWord, [endWord])
```

### 675. Cut Off Trees for Golf Event

```python
class Solution:
    def cutOffTree(self, forest: List[List[int]]) -> int:
        def bfs(startX, startY, targetX, targetY):
            queue = deque([(startX, startY, 0)])
            visited = set([(startX, startY)])
            while queue:
                r, c, steps = queue.popleft()
                if r == targetX and c == targetY:
                    return steps
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C and forest[row][col] != 0 and (row, col) not in visited:
                        queue.append((row, col, steps + 1))
                        visited.add((row, col))
            return -1

        R, C = len(forest), len(forest[0])
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        trees = sorted([(forest[r][c], r, c) for r in range(R) for c in range(C) if forest[r][c] > 1])
        res = 0
        trees = [(0, 0, 0)] + trees
        for a, b in pairwise(trees):
            steps = bfs(a[1], a[2], b[1], b[2])
            if steps == -1:
                return -1
            res += steps
        return res
```