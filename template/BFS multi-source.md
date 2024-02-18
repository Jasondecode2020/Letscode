## tempalte: multi-source

- 1 find multi-source
- 2 prepare q or visited set
- 3 normal bfs using template 2


## template: Multi-source BFS

* `286. Walls and Gates`
* `542. 01 Matrix`
* `815. Bus Routes`
* `994. Rotting Oranges`
* `934. Shortest Bridge`
* `1020. Number of Enclaves`
* `1162. As Far from Land as Possible`
* `773. Sliding Puzzle`
* `733. Flood Fill`
* `2684. Maximum Number of Moves in a Grid`
* `1926. Nearest Exit from Entrance in Maze`



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

### 542. 01 Matrix

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        q, visited = deque(), set()
        for r in range(R):
            for c in range(C):
                if mat[r][c] == 0:
                    q.append((r, c, 0))
                    visited.add((r, c))
        dp = [[0] * C for r in range(R)]
        while q:
            r, c, n = q.popleft()
            dp[r][c] = n 
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited:
                    visited.add((row, col))
                    q.append((row, col, n + 1))
        return dp
```

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

### 1293. Shortest Path in a Grid with Obstacles Elimination

```python
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        pq = [(0, k, 0, 0)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited = set([(0, 0, k)])
        if k >= R + C - 3: # trick, k equal to R + C - 3, return 
            return R + C - 2
        while pq:
            d, threshold, r, c = heappop(pq)
            if r == R - 1 and c == C - 1:
                return d
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and (row, col, threshold) not in visited:
                    if grid[row][col] == 0 and threshold >= 0:
                        heappush(pq, (d + 1, threshold, row, col))
                        visited.add((row, col, threshold))
                    else:
                        if threshold >= 1:
                            heappush(pq, (d + 1, threshold - 1, row, col))
                            visited.add((row, col, threshold - 1))
        return -1
```


### 490. The Maze

```python
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        def check(r, c, dr, dc):
            while 0 <= r < R and 0 <= c < C and maze[r][c] != 1:
                r += dr 
                c += dc 
            return r - dr, c - dc

        q = deque([tuple(start)])
        visited = set([tuple(start)])
        R, C = len(maze), len(maze[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while q:
            r, c = q.popleft()
            if r == destination[0] and c == destination[1]:
                return True
            for dr, dc in directions:
                row, col = check(r, c, dr, dc)
                if (row, col) not in visited:
                    visited.add((row, col))
                    q.append((row, col))
        return False
```

### 773. Sliding Puzzle

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        def check(s, i, r, c):
            idx = r * 3 + c 
            s = list(s)
            s[idx], s[i] = s[i], s[idx]
            return ''.join(s)

        s = ''.join(str(n) for row in board for n in row)
        R, C, direction = 2, 3, [[0, 1], [0, -1], [1, 0], [-1, 0]]
        q, visited = deque([(s, 0)]), set([s])
        while q:
            pattern, step = q.popleft()
            if pattern == '123450':
                return step
            idx = pattern.index('0')
            r, c = idx // 3, idx % 3
            for dr, dc in direction:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C:
                    new_pattern = check(pattern, idx, row, col)
                    if new_pattern not in visited:
                        visited.add(new_pattern)
                        q.append((new_pattern, step + 1))
        return -1
```

### 733. Flood Fill

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        q = deque([(sr, sc)])
        R, C = len(image), len(image[0])
        visited = set([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and image[row][col] == image[r][c]:
                    visited.add((row, col))
                    q.append((row, col))
        for r, c in visited:
            image[r][c] = color 
        return image
```

### 2684. Maximum Number of Moves in a Grid

```python
class Solution:
    def maxMoves(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        q = deque([(r, 0, 0) for r in range(R)])
        visited = set([(r, 0) for r in range(R)])
        
        while q:
            r, c, count = q.popleft()
            for row, col in [(r - 1, c + 1), (r, c + 1), (r + 1, c + 1)]:
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and grid[row][col] > grid[r][c]:
                    q.append((row, col, count + 1))
                    visited.add((row, col))
        return count
```

### 1926. Nearest Exit from Entrance in Maze

```python
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        R, C = len(maze), len(maze[0])
        exit = set()
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and maze[r][c] == '.':
                    exit.add((r, c))
        if tuple(entrance) in exit:
            exit.remove(tuple(entrance))

        q, visited = deque([(entrance[0], entrance[1], 0)]), set([tuple(entrance)])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while q:
            r, c, d = q.popleft()
            if (r, c) in exit:
                return d 
            for dr, dc in directions:
                x, y = r + dr, c + dc 
                if 0 <= x < R and 0 <= y < C and (x, y) not in visited and maze[x][y] == '.':
                    visited.add((x, y))
                    q.append((x, y, d + 1))
        return -1
```

### 1162. As Far from Land as Possible

```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        q = deque()
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        R, C = len(grid), len(grid[0])
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    q.append((r, c, 0))

        res = 0
        while q:
            r, c, t = q.popleft()
            res = max(res, t)
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 0:
                    q.append((row, col, t + 1))
                    grid[row][col] = 2
        return res if res != 0 else -1
```
