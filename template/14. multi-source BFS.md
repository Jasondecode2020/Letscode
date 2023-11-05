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

## tempalte: multi-source

- 1 find multi-source
- 2 prepare q or visited set
- 3 normal bfs using template 2


## Vinilla BFS

- 815. Bus Routes

## template: Multi-source BFS

* `994. Rotting Oranges`
* `286. Walls and Gates`
* `934. Shortest Bridge`
* `286. Walls and Gates`
* `1020. Number of Enclaves`

### 815. Bus Routes

```python
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        g = defaultdict(list) # stop: [bus]
        for bus, route in enumerate(routes):
            for stop in route:
                g[stop].append(bus)
        
        q = deque([(source, 0)])
        buses = [set(r) for r in routes]
        visited_bus, visited_stop = set(), set([source])
        while q:
            stop, cost = q.popleft()
            if stop == target:
                return cost
            for bus in g[stop]:
                if bus not in visited_bus:
                    visited_bus.add(bus)
                    for s in buses[bus]:
                        if s not in visited_stop:
                            visited_stop.add(s)
                            q.append((s, cost + 1))
        return -1
```

### 994. Rotting Oranges

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        R, C = len(grid), len(grid[0])
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 2:
                    q.append((r, c, 0))

        res = 0
        while q:
            r, c, t = q.popleft()
            res = max(res, t)
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) and grid[row][col] == 1:
                    q.append((row, col, t + 1))
                    grid[row][col] = 2

        if any(1 in r for r in grid):
            return -1
        return res
```

### 934. Shortest Bridge

```python
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
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
                        step, row, col = heappop(pq)
                        for dr, dc in directions:
                            x, y = row + dr, col + dc 
                            if 0 <= x < R and 0 <= y < C :
                                if grid[x][y] == 1:
                                    return step
                                if grid[x][y] == 0:
                                    grid[x][y] = 2
                                    heappush(pq, (step + 1, x, y))
                        
```

### 286. Walls and Gates

```python
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        R, C, INF = len(rooms), len(rooms[0]), 2147483647
        q = deque()
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for r in range(R):
            for c in range(C):
                if rooms[r][c] == 0:
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and rooms[row][col] == INF:
                    q.append((row, col))
                    rooms[row][col] = rooms[r][c] + 1
        return rooms
```

### 1020. Number of Enclaves

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        q = deque()
        R, C = len(grid), len(grid[0])
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and grid[r][c] == 1:
                    q.append((r, c))
                    grid[r][c] = 0
        
        direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while q:
            r, c = q.popleft()
            for dr, dc in direction:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    grid[row][col] = 0
                    q.append((row, col))
        return sum(grid[r][c] for r in range(R) for c in range(C))
```