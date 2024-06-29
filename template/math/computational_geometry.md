## Question list(14)

* [812. Largest Triangle Area](#812-largest-triangle-area)
* [836. Rectangle Overlap](#836-rectangle-overlap)
* [223. Rectangle Area](#223-rectangle-area)
* [2481. Minimum Cuts to Divide a Circle](#2481-minimum-cuts-to-divide-a-circle)
* [1232. Check If It Is a Straight Line](#1232-check-if-it-is-a-straight-line)

* [1266. Minimum Time Visiting All Points](#1266-minimum-time-visiting-all-points)
* [1037. Valid Boomerang](#1037-valid-boomerang)
* [883. Projection Area of 3D Shapes](#883-projection-area-of-3d-shapes)
* [892. Surface Area of 3D Shapes](#892-surface-area-of-3d-shapes)
* [1030. Matrix Cells in Distance Order](#1030-matrix-cells-in-distance-order)

* [469. Convex Polygon](#469-convex-polygon)
* [356. Line Reflection](#356-line-reflection)
* [593. Valid Square](#593-valid-square)
* [2613. Beautiful Pairs](#2613-beautiful-pairs)

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

### 593. Valid Square

```python
class Solution:
    def validSquare(self, p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
        def check(a, b, c, d):
            x1, y1 = a
            x2, y2 = b
            x3, y3 = c 
            x4, y4 = d 
            side1 = (x2 - x1) ** 2 + (y2 - y1) ** 2
            side2 = (x3 - x2) ** 2 + (y3 - y2) ** 2
            side3 = (x4 - x3) ** 2 + (y4 - y3) ** 2
            side4 = (x4 - x1) ** 2 + (y4 - y1) ** 2
            diag1 = (x3 - x1) ** 2 + (y3 - y1) ** 2
            diag2 = (x2 - x4) ** 2 + (y2 - y4) ** 2
            if side1 and side1 == side2 == side3 == side4 and diag1 and diag1 == diag2:
                return True
            return False
        for a, b, c, d in permutations([p1, p2, p3, p4]):
            if check(a, b, c, d):
                return True
        return False
```

### 2613. Beautiful Pairs

```python
class Solution:
    def beautifulPair(self, nums1: List[int], nums2: List[int]) -> List[int]:
        def m_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)

        def dfs(l, r):
            if l >= r:
                return inf, -1, -1
          
            m = l + (r - l) // 2
            x = points[m][0]
            d_l, l_i, l_j = dfs(l, m)
            d_r, r_i, r_j = dfs(m + 1, r)
            if d_l > d_r or (d_l == d_r and (l_i > r_i or (l_i == r_i and l_j > r_j))):
                d_l, l_i, l_j = d_r, r_i, r_j
        
            candidates = [p for p in points[l : r + 1] if abs(p[0] - x) <= d_l]
            candidates.sort(key=lambda p: p[1])
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if candidates[j][1] - candidates[i][1] > d_l:
                        break
                    index_i, index_j = sorted([candidates[i][2], candidates[j][2]])
                    dist = m_distance(candidates[i][0], candidates[i][1], candidates[j][0], candidates[j][1])
                    if dist < d_l or (dist == d_l and (index_i < l_i or (index_i == l_i and index_j < l_j))):
                        d_l, l_i, l_j = dist, index_i, index_j
            return d_l, l_i, l_j

        d = defaultdict(list)
        for i, (x, y) in enumerate(zip(nums1, nums2)):
            d[(x, y)].append(i)
      
        points = []
        for i, (x, y) in enumerate(zip(nums1, nums2)):
            if len(d[(x, y)]) > 1:
                return [i, d[(x, y)][1]]
            points.append((x, y, i))
      
        points.sort()
        dist, index1, index2 = dfs(0, len(points) - 1)
        return [index1, index2]
```