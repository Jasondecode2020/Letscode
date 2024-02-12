## template

```python
def fn(arr):
    @cache
    def dfs(STATE):
        if BASE_CASE:
            return 0
        ans = RECURRENCE_RELATION(STATE)
        return ans
    return dfs(INPUT)
```

### 329. Longest Increasing Path in a Matrix

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        @cache
        def dfs(r, c):
            res = 1
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if  0 <= row < R and 0 <= col < C and matrix[row][col] > matrix[r][c]:
                    res = max(res, dfs(row, col) + 1)
            return res

        res = -inf
        for r in range(R):
            for c in range(C):
                res = max(res, dfs(r, c))
        return res
```

### 2328. Number of Increasing Paths in a Grid

```python
class Solution:
    def countPaths(self, matrix: List[List[int]]) -> int:
        R, C, res = len(matrix), len(matrix[0]), 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        mod = 10 ** 9 + 7
        @cache
        def dfs(r, c):
            res = 1
            for dr, dc in directions:
                row, col = dr + r, dc + c 
                if 0 <= row < R and 0 <= col < C and matrix[row][col] > matrix[r][c]:
                    res += dfs(row, col)
            return res
        
        for r in range(R):
            for c in range(C):
                res += dfs(r, c)
        return res % mod
```

### 1186. Maximum Subarray Sum with One Deletion

```python
class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        n = len(arr)
        @cache
        def dfs(i, sign):
            if i < 0:
                return -inf 
            if sign == 0:
                return max(dfs(i - 1, 0), 0) + arr[i]
            return max(dfs(i - 1, 1) + arr[i], dfs(i - 1, 0))
        return max(max(dfs(i, 0), dfs(i, 1)) for i in range(n))
```