## Template: 0-1 BFS

### Question list

* `2290. Minimum Obstacle Removal to Reach Corner`
* `1368. Minimum Cost to Make at Least One Valid Path in a Grid`
* `1293. Minimum Obstacle Removal to Reach Corner`

### 934. Shortest Bridge

- 0-1 BFS

```python
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dist = [[inf] * C for r in range(R)]
        dist[0][0] = 0
        q = deque([(0, 0)])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C:
                    obstacle = grid[r][c]
                    if obstacle + dist[r][c] < dist[row][col]:
                        dist[row][col] = obstacle + dist[r][c]
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
        dist = [[inf] * C for r in range(R)]
        dist[0][0] = 0
        q = deque([(0, 0)])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C:
                    obstacle = grid[r][c]
                    if obstacle + dist[r][c] < dist[row][col]:
                        dist[row][col] = obstacle + dist[r][c]
                        if not obstacle:
                            q.appendleft((row, col))
                        else:
                            q.append((row, col))
        return dist[-1][-1]
```

### 1368. Minimum Cost to Make at Least One Valid Path in a Grid

- 0-1 BFS

```python
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dist = [[inf] * C for r in range(R)]
        dist[0][0] = 0
        q = deque([(0, 0)])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]] # right, left, down, up
        while q:
            r, c = q.popleft()
            for i, (dr, dc) in enumerate(directions):
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C:
                    obstacle = i + 1 != grid[r][c]
                    if obstacle + dist[r][c] < dist[row][col]:
                        dist[row][col] = obstacle + dist[r][c]
                        if not obstacle:
                            q.appendleft((row, col))
                        else:
                            q.append((row, col))
        return dist[-1][-1]
```