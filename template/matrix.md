## matrix

```python
def fn():
    R, C = len(img), len(img[0])
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]] # 4 direction
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]] # 8 direction
```

### 304. Range Sum Query 2D - Immutable

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.mat = matrix
        R, C = len(matrix), len(matrix[0])
        for r in range(1, R):
            self.mat[r][0] += self.mat[r - 1][0]
        for c in range(1, C):
            self.mat[0][c] += self.mat[0][c - 1]
        for r in range(1, R):
            for c in range(1, C):
                self.mat[r][c] += self.mat[r - 1][c] + self.mat[r][c - 1] - self.mat[r - 1][c - 1]
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        left = self.mat[row2][col1 - 1] if col1 - 1 >= 0 else 0
        top = self.mat[row1 - 1][col2] if row1 - 1 >= 0 else 0
        topLeft = self.mat[row1 - 1][col1 - 1] if row1 >= 1 and col1 >= 1 else 0
        return self.mat[row2][col2] - left - top + topLeft



# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```


### 422. Valid Word Square

```python
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        R, C = len(words), max(len(w) for w in words)
        if R != C:
            return False
        m = [[''] * C for r in range(R)]
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                m[i][j] = c 
        for i in range(R):
            for j in range(i + 1, C):
                if m[i][j] != m[j][i]:
                    return False
        return True
```

### 661. Image Smoother

```python
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        R, C = len(img), len(img[0])
        direction = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        dp = deepcopy(img)
        for r in range(R):
            for c in range(C):
                count, total = 1, img[r][c]
                for dr, dc in direction:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C:
                        count += 1
                        total += img[row][col]
                dp[r][c] = total // count
        return dp
```

### 2319. Check if Matrix Is X-Matrix

```python
class Solution:
    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        R, C = len(grid), len(grid[0])
        for r in range(R):
            for c in range(C):
                if r == c or r + c + 1 == R:
                    if grid[r][c] == 0:
                        return False
                elif grid[r][c] != 0:
                    return False
        return True
```

### 2643. Row With Maximum Ones

```python
class Solution:
    def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
        idx, count = 0, 0
        for i, m in enumerate(mat):
            v = m.count(1)
            if v > count:
                idx, count = i, v 
        return [idx, count]
```

### 73. Set Matrix Zeroes

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        R, C, row, col = len(matrix), len(matrix[0]), set(), set()
        for r in range(R):
            for c in range(C):
                if matrix[r][c] == 0:
                    row.add(r)
                    col.add(c)
        for r in range(R):
            for c in range(C):
                if r in row or c in col:
                    matrix[r][c] = 0
```

### 1886. Determine Whether Matrix Can Be Obtained By Rotation

```python
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        R, C = len(mat), len(mat[0])
        def rotate():
            mat.reverse()
            for r in range(R):
                for c in range(r + 1, C):
                    mat[r][c], mat[c][r] = mat[c][r], mat[r][c]
        def equal():
            for a, b in zip(mat, target):
                if a != b:
                    return False
            return True
        for i in range(4):
            if equal():
                return True
            rotate()
        return False
```

### 750. Number Of Corner Rectangles

```python
class Solution:
    def countCornerRectangles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        d = Counter()
        for row in grid:
            for c1 in range(C):
                for c2 in range(c1 + 1, C):
                    if row[c1] and row[c2]:
                        d[(c1, c2)] += 1

        res = 0
        for n in d.values():
            res += n * (n - 1) // 2
        return res
```

### 835. Image Overlap

```python
class Solution:
    def largestOverlap(self, img1: List[List[int]], img2: List[List[int]]) -> int:
        R, C = len(img1), len(img1[0])
        def check(img):
            arr = []
            for r in range(R):
                for c in range(C):
                    if img[r][c] == 1:
                        arr.append((r, c))
            return arr

        arr1, arr2 = check(img1), check(img2)
        d = defaultdict(int)
        res = 0
        for x1, y1 in arr1:
            for x2, y2 in arr2:
                translation = (x2 - x1, y2 - y1)
                d[translation] += 1
                res = max(res, d[translation])
        return res
```

### 59. Spiral Matrix II

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        m, n = n, n 
        matrix = [[-1] * n for r in range(m)]
        direction = 'left'
        r, c = 0, 0
        top, down, left, right = 0, m, 0, n
        val = 1
        while val < n * n + 1:
            matrix[r][c] = val 
            if direction == 'left':
                if c + 1 < right:
                    c += 1
                else:
                    r += 1
                    top += 1
                    direction = 'down'
            elif direction == 'down':
                if r + 1 < down:
                    r += 1
                else:
                    c -= 1
                    right -= 1
                    direction = 'right'
            elif direction == 'right':
                if c - 1 >= left:
                    c -= 1
                else:
                    r -= 1
                    down -= 1
                    direction = 'top'
            elif direction == 'top':
                if r - 1 >= top:
                    r -= 1
                else:
                    c += 1
                    left += 1
                    direction = 'left'
            val += 1
        return matrix
```

### 2326. Spiral Matrix IV

```python
class Solution:
    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
        matrix = [[-1] * n for r in range(m)]
        direction = 'left'
        r, c = 0, 0
        top, down, left, right = 0, m, 0, n
        while head:
            val = head.val
            head = head.next 
            matrix[r][c] = val 
            if direction == 'left':
                if c + 1 < right:
                    c += 1
                else:
                    r += 1
                    top += 1
                    direction = 'down'
            elif direction == 'down':
                if r + 1 < down:
                    r += 1
                else:
                    c -= 1
                    right -= 1
                    direction = 'right'
            elif direction == 'right':
                if c - 1 >= left:
                    c -= 1
                else:
                    r -= 1
                    down -= 1
                    direction = 'top'
            elif direction == 'top':
                if r - 1 >= top:
                    r -= 1
                else:
                    c += 1
                    left += 1
                    direction = 'left'
        return matrix
```

### 1329. Sort the Matrix Diagonally

```python
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        d = defaultdict(list)
        for r in range(R):
            for c in range(C):
                d[r - c].append((mat[r][c], r, c))
        for a in d.values():
            a.sort()
            v = [n[0] for n in a]
            coodinates = sorted([(x, y) for v, x, y in a])
            for n, (x, y) in zip(v, coodinates):
                mat[x][y] = n 
        return mat
```

### 498. Diagonal Traverse

```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        R, C = len(mat), len(mat[0])
        d = defaultdict(list)
        for r in range(R):
            for c in range(C):
                d[r + c].append(mat[r][c])
        
        res = []
        for i in range(R + C - 1):
            if i % 2 == 0:
                res.extend(d[i][::-1])
            else:
                res.extend(d[i])
        return res
```

### 885. Spiral Matrix III

```python 
class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        res = [(rStart, cStart)]
        if rows * cols == 1:
            return res 
        
        for k in range(1, 2 * max(rows, cols), 2):
            for dr, dc, dk in [(0, 1, k), (1, 0, k), (0, -1, k + 1), (-1, 0, k + 1)]:
                for _ in range(dk):
                    rStart += dr
                    cStart += dc 
                    if 0 <= rStart < rows and 0 <= cStart < cols:
                        res.append((rStart, cStart))
                        if len(res) == rows * cols:
                            return res 
```