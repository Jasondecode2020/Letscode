## Area of triangle

```python
def area(a, b, c):
    s = (a + b + c) / 2
    return sqrt(s * (s - a) * (s - b) * (s - c)) if (s - a) >= 0 and (s - b) >= 0 and (s - c) >= 0 else 0
```


### 812. Largest Triangle Area

```python
class Solution:
    def largestTriangleArea(self, points: List[List[int]]) -> float:
        def area(a, b, c):
            s = (a + b + c) / 2
            return sqrt(s * (s - a) * (s - b) * (s - c)) if (s - a) >= 0 and (s - b) >= 0 and (s - c) >= 0 else 0

        n, res = len(points), 0
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    a = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    b = sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                    c = sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
                    res = max(res, area(a, b, c))
        return res
```

### 836. Rectangle Overlap

- similar to 836

```python
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        x1, y1, x2, y2 = rec1
        x3, y3, x4, y4 = rec2
        if y3 >= y2 or y1 >= y4 or x3 >= x2 or x1 >= x4:
            return False
        return True
```

### 223. Rectangle Area

- similar to 223

```python
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        x1, y1, x2, y2 = ax1, ay1, ax2, ay2
        x3, y3, x4, y4 = bx1, by1, bx2, by2
        overlap = (min(x2, x4) - max(x1, x3)) * (min(y2, y4) - max(y1, y3))
        first = (x2 - x1) * (y2 - y1)
        second = (x4 - x3) * (y4 - y3)
        if y3 >= y2 or y1 >= y4 or x3 >= x2 or x1 >= x4:
            return first + second
        return first + second - overlap
```

### 2481. Minimum Cuts to Divide a Circle

```python
class Solution:
    def numberOfCuts(self, n: int) -> int:
        if n == 1:
            return 0
        return n // 2 if n % 2 == 0 else n
```

### 1232. Check If It Is a Straight Line

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x, y = x2 - x1, y2 - y1
        for i in range(2, len(coordinates)):
            u, v = coordinates[i][0] - x1, coordinates[i][1] - y1
            if x * v - y * u:
                return False
        return True
```

### 1266. Minimum Time Visiting All Points

```python
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        res = 0
        n = len(points)
        if n == 1:
            return 0
        for i in range(1, n):
            res += max(abs(points[i][0] - points[i - 1][0]), abs(points[i][1] - points[i - 1][1]))
        return res
```

### 1037. Valid Boomerang

```python
class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        if len(points) != len(set([(x, y) for x, y in points])):
            return False
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x, y = x2 - x1, y2 - y1
        u, v = x3 - x1, y3 - y1
        if x * v - y * u == 0:
            return False
        return True
```

### 883. Projection Area of 3D Shapes

```python
class Solution:
    def projectionArea(self, grid: List[List[int]]) -> int:
        # [1,2]
        # [3,4]
        left = sum(max(arr) for arr in grid)
        down = sum(grid[r][c] > 0 for r in range(len(grid)) for c in range(len(grid[0])))
        front = sum(max(arr) for arr in zip(*grid))
        return left + down + front
```

### 883. Projection Area of 3D Shapes

```python
class Solution:
    def projectionArea(self, grid: List[List[int]]) -> int:
        n = len(grid)
        left = sum(max(arr) for arr in grid)
        down = sum(grid[r][c] > 0 for r in range(n) for c in range(n))
        front = sum(max(arr) for arr in zip(*grid))
        return left + down + front
```

### 892. Surface Area of 3D Shapes

```python
class Solution:
    def surfaceArea(self, grid: List[List[int]]) -> int:
        n = len(grid) + 2
        padding = [[0] * n for r in range(n)]
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                padding[r][c] = grid[r - 1][c - 1]

        res = 0
        R, C = n, n
        for r in range(1, R - 1):
            for c in range(1, C - 1):
                res += padding[r][c] * 4 + 2 if padding[r][c] else padding[r][c] * 4
                # top
                if padding[r][c] > padding[r - 1][c]:
                    res -= padding[r - 1][c]
                else:
                    res -= padding[r][c]
                # bottom
                if padding[r][c] > padding[r + 1][c]:
                    res -= padding[r + 1][c]
                else:
                    res -= padding[r][c]
                # left
                if padding[r][c] > padding[r][c - 1]:
                    res -= padding[r][c - 1]
                else:
                    res -= padding[r][c]
                # right
                if padding[r][c] > padding[r][c + 1]:
                    res -= padding[r][c + 1]
                else:
                    res -= padding[r][c]
        return res
```

### 1030. Matrix Cells in Distance Order

```python
class Solution:
    def allCellsDistOrder(self, rows: int, cols: int, rCenter: int, cCenter: int) -> List[List[int]]:
        res = []
        for r in range(rows):
            for c in range(cols):
                res.append((abs(r - rCenter) + abs(c - cCenter), r, c))
        res.sort()
        return [[r, c] for d, r, c in res]
```

### 469. Convex Polygon

- cross product

```python
class Solution:
    def isConvex(self, points: List[List[int]]) -> bool:
        def cross(A, B, C):
            x, y = B[0] - A[0], B[1] - A[1]
            u, v = C[0] - A[0], C[1] - A[1]
            return x * v - y * u

        prev = 0
        n = len(points)
        for i in range(n):
            # need to check all points
            cur = cross(points[i], points[(i + 1) % n], points[(i + 2) % n]) 
            if cur != 0: # if cur = 0 means in one line, no need to check
                if cur * prev < 0:
                    return False
                else:
                    prev = cur
        return True
```

### 356. Line Reflection

```python
class Solution:
    def isReflected(self, points: List[List[int]]) -> bool:
        mn_x, mx_x = min(p[0] for p in points), max(p[0] for p in points)
        point = (mn_x + mx_x) / 2
        c = Counter()
        for x, y in points:
            c[(x, y)] += 1
        for x, y in c:
            if c[(2 * point - x, y)] == 0:
                return False
        return True
```