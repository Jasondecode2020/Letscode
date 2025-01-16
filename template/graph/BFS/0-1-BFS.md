## Template: 0-1 BFS


```python
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dist = [[inf] * C for _ in range(R)]
        dist[0][0] = 0
        q = deque([(0, 0)])
        while q:
            r, c = q.popleft()
            for i, (dr, dc) in enumerate(directions):
                obstacle = i + 1 != grid[r][c]
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C:
                    if dist[r][c] + obstacle < dist[row][col]:
                        dist[row][col] = dist[r][c] + obstacle
                        if not obstacle:
                            q.appendleft((row, col))
                        else:
                            q.append((row, col))
        return dist[-1][-1]
```

### Question list

* [1368. Minimum Cost to Make at Least One Valid Path in a Grid](#1368-minimum-cost-to-make-at-least-one-valid-path-in-a-grid)
* [2290. Minimum Obstacle Removal to Reach Corner](#2290-minimum-obstacle-removal-to-reach-corner)
* [934. Shortest Bridge](#934-shortest-bridge)
* [3286. Find a Safe Walk Through a Grid](#3286-find-a-safe-walk-through-a-grid)

### 1368. Minimum Cost to Make at Least One Valid Path in a Grid

- 0-1 BFS

```python
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dist = [[inf] * C for _ in range(R)]
        dist[0][0] = 0
        q = deque([(0, 0)])
        while q:
            r, c = q.popleft()
            for i, (dr, dc) in enumerate(directions):
                obstacle = i + 1 != grid[r][c]
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C:
                    if dist[r][c] + obstacle < dist[row][col]:
                        dist[row][col] = dist[r][c] + obstacle
                        if not obstacle:
                            q.appendleft((row, col))
                        else:
                            q.append((row, col))
        return dist[-1][-1]
```


### 2290. Minimum Obstacle Removal to Reach Corner

- normal BFS

```python
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        steps = 1 if grid[0][0] else 0
        pq, visited = [(steps, 0, 0)], set([(0, 0)])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while pq:
            step, r, c = heappop(pq)
            if r == R - 1 and c == C - 1:
                return step
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited:
                    visited.add((row, col))
                    if grid[row][col]:
                        heappush(pq, (step + 1, row, col))
                    else:
                        heappush(pq, (step, row, col))
```

- 0-1 BFS

```python
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dist = [[inf] * C for _ in range(R)]
        dist[0][0] = 0
        q = deque([(0, 0)])
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C:
                    if dist[r][c] + grid[row][col] < dist[row][col]:
                        dist[row][col] = dist[r][c] + grid[row][col]
                        if not grid[row][col]:
                            q.appendleft((row, col))
                        else:
                            q.append((row, col))
        return dist[-1][-1]
```

### 934. Shortest Bridge

- 0-1 BFS

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        visited = [[False] * C for _ in range(R)]
        def bfs(r, c):
            q = deque([(r, c, 0)])
            while q:
                r, c, flips = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and not visited[row][col]:
                        visited[row][col] = True
                        if grid[row][col] == 1:
                            if flips >= 1:
                                return flips
                            q.appendleft((row, col, 0))
                        else:
                            q.append((row, col, flips + 1))
        
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    visited[r][c] = True
                    return bfs(r, c)
```

### 3286. Find a Safe Walk Through a Grid

```python
class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        R, C = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dist = [[inf] * C for _ in range(R)]
        dist[0][0] = 0 if grid[0][0] == 0 else 1
        q = deque([(0, 0)])
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C:
                    if dist[r][c] + grid[row][col] < dist[row][col]:
                        dist[row][col] = dist[r][c] + grid[row][col]
                        if not grid[row][col]:
                            q.appendleft((row, col))
                        else:
                            q.append((row, col))
        return dist[-1][-1] <= health - 1
```