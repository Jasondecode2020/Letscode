

## template: dfs grid

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
* [529. Minesweeper](#529-minesweeper)
* [827. Making A Large Island](#827-making-a-large-island)
* [1905. Count Sub Islands](#1905-count-sub-islands)
* [1559. Detect Cycles in 2D Grid](#1559-detect-cycles-in-2d-grid)

## Grid BFS

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
* [1905. Count Sub Islands]()
* [1559. Detect Cycles in 2D Grid](#1559-detect-cycles-in-2d-grid)

* [542. 01 Matrix](#542-01-matrix)
* [994. Rotting Oranges](#994-rotting-oranges)
* [2684. Maximum Number of Moves in a Grid](#2684-maximum-number-of-moves-in-a-grid) 1626
* [1926. Nearest Exit from Entrance in Maze](#1926-nearest-exit-from-entrance-in-maze) 1638
* [1162. As Far from Land as Possible](#1162-as-far-from-land-as-possible) 1666
* [934. Shortest Bridge]() 1826
2146. 价格范围内最高排名的 K 样物品 1837
* [1293. Shortest Path in a Grid with Obstacles Elimination]()
1210. 穿过迷宫的最少移动次数 2022
317. 离建筑物最近的距离（会员题）
* [675. Cut Off Trees for Golf Event](#675-cut-off-trees-for-golf-event)

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