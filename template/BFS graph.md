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