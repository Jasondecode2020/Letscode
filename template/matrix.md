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