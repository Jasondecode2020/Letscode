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