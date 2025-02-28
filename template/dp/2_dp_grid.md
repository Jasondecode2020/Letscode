## 2 Grid (23)

### 2.1 Basics (10)

* [62. Unique Paths](#62-unique-paths)
* [63. Unique Paths II](#63-unique-paths-ii)
* [64. Minimum Path Sum](#64-minimum-path-sum)
* [120. Triangle](#120-triangle)
* [3393. Count Paths With the Given XOR Value](#3393-count-paths-with-the-given-xor-value)
* [931. Minimum Falling Path Sum](#931-minimum-falling-path-sum)
* [2684. Maximum Number of Moves in a Grid](#2684-maximum-number-of-moves-in-a-grid)
* [1289. Minimum Falling Path Sum II](#1289-minimum-falling-path-sum-ii)
* [2304. Minimum Path Cost in a Grid](#2304-minimum-path-cost-in-a-grid)
* [3418. Maximum Amount of Money Robot Can Earn](#3418-maximum-amount-of-money-robot-can-earn)

### 2.2 Advanced (13)

* [1594. Maximum Non Negative Product in a Matrix](#1594-maximum-non-negative-product-in-a-matrix)
* [1301. Number of Paths with Max Score](#1301-number-of-paths-with-max-score)
* [2435. Paths in Matrix Whose Sum Is Divisible by K](#2435-paths-in-matrix-whose-sum-is-divisible-by-k) 1951
* [174. Dungeon Game](#174-dungeon-game)
* [329. Longest Increasing Path in a Matrix](#329-longest-increasing-path-in-a-matrix)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2267. Check if There Is a Valid Parentheses String Path](#2267-check-if-there-is-a-valid-parentheses-string-path)
* [2510. Check if There is a Path With Equal Number of 0's And 1's](#2510-check-if-there-is-a-path-with-equal-number-of-0s-and-1s)
* [1937. Maximum Number of Points with Cost](#1937-maximum-number-of-points-with-cost)
* [1463. Cherry Pickup II](#1463-cherry-pickup-ii)
* [741. Cherry Pickup](#741-cherry-pickup)
* [3363. Find the Maximum Number of Fruits Collected](#3363-find-the-maximum-number-of-fruits-collected) 2400
* [3459. Length of Longest V-Shaped Diagonal Segment](#3459-length-of-longest-v-shaped-diagonal-segment) 2500

### 62. Unique Paths

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        R, C = m, n
        dp = [[1] * C for i in range(R)]
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
        return dp[-1][-1]
```

### 63. Unique Paths II

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        R, C = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[1] * C for i in range(R)]
        if obstacleGrid[0][0] == 1: return 0
        for c in range(1, C):
            dp[0][c] = dp[0][c - 1] if obstacleGrid[0][c] != 1 else 0
        for r in range(1, R):
            dp[r][0] = dp[r - 1][0] if obstacleGrid[r][0] != 1 else 0
        for r in range(1, R):
            for c in range(1, C):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1] if obstacleGrid[r][c] != 1 else 0
        return dp[-1][-1]
```

### 64. Minimum Path Sum

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        for r in range(1, R):
            grid[r][0] += grid[r - 1][0]
        for c in range(1, C):
            grid[0][c] += grid[0][c - 1]
        for r in range(1, R):
            for c in range(1, C):
                grid[r][c] += min(grid[r - 1][c], grid[r][c - 1])
        return grid[-1][-1]
```

### 120. Triangle

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        for r in range(n - 2, -1, -1):
            for c in range(r + 1):
                triangle[r][c] += min(triangle[r + 1][c: c + 2])
        return triangle[0][0]
```

### 3393. Count Paths With the Given XOR Value

```python 
class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        mod = 10 ** 9 + 7 
        @cache
        # def f(r, c, x):
        #     if r < 0 or c < 0:
        #         return 0 
        #     val = grid[r][c]
        #     if r == 0 and c == 0:
        #         return 1 if x == val else 0
        #     return (f(r, c - 1, x ^ val) + f(r - 1, c, x ^ val)) % mod 
        # return f(R - 1, C - 1, k)
        @cache
        def f(r, c, x):
            if r == R or c == C:
                return 0 
            x ^= grid[r][c]
            if r == R - 1 and c == C - 1:
                return 1 if x == k else 0
            return (f(r, c + 1, x) + f(r + 1, c, x)) % mod 
        return f(0, 0, 0)
```

### 931. Minimum Falling Path Sum

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0]) + 2
        dp = [[inf] * C for i in range(R)]
        for c in range(1, C - 1):
            dp[0][c] = matrix[0][c - 1]
        for r in range(1, R):
            for c in range(1, C - 1):
                dp[r][c] = min(dp[r - 1][c - 1: c + 2]) + matrix[r][c - 1]
        return min(dp[-1])
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

### 1289. Minimum Falling Path Sum II

```python
class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        for r in range(1, R):
            for c in range(C):
                matrix[r][c] += min(matrix[r - 1][:c] + matrix[r - 1][c + 1:])
        return min(matrix[-1])
```

### 2304. Minimum Path Cost in a Grid

```python
class Solution:
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        for r in range(R - 2, -1, -1):
            for c in range(C):
                grid[r][c] += min(n + c for n, c in zip(grid[r + 1], moveCost[grid[r][c]]))
        return min(grid[0])
```

### 3418. Maximum Amount of Money Robot Can Earn

```python
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        R, C = len(coins), len(coins[0])
        @cache
        def f(r, c, k):
            if r < 0 or c < 0:
                return -inf 
            x = coins[r][c]
            if r == 0 and c == 0:
                return max(x, 0) if k else x 
            res = max(f(r - 1, c, k), f(r, c - 1, k)) + coins[r][c]
            if k and x < 0:
                res = max(res, f(r - 1, c, k - 1), f(r, c - 1, k - 1))
            return res 
        return f(R - 1, C - 1, 2)
```

### 1594. Maximum Non Negative Product in a Matrix

```python
class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        R, C = len(grid), len(grid[0])
        mx_grid, mn_grid = deepcopy(grid), deepcopy(grid)
        for c in range(1, C):
            mx_grid[0][c] = mn_grid[0][c] = mx_grid[0][c - 1] * grid[0][c]
        for r in range(1, R):
            mx_grid[r][0] = mn_grid[r][0] = mx_grid[r - 1][0] * grid[r][0]
        print(mx_grid, mn_grid)
        for r in range(1, R):
            for c in range(1, C):
                if grid[r][c] >= 0:
                    mx_grid[r][c] = max(mx_grid[r - 1][c], mx_grid[r][c - 1]) * grid[r][c]
                    mn_grid[r][c] = min(mn_grid[r - 1][c], mn_grid[r][c - 1]) * grid[r][c]
                else:
                    mx_grid[r][c] = min(mn_grid[r - 1][c], mn_grid[r][c - 1]) * grid[r][c]
                    mn_grid[r][c] = max(mx_grid[r - 1][c], mx_grid[r][c - 1]) * grid[r][c]
        return mx_grid[-1][-1] % mod if mx_grid[-1][-1] >= 0 else -1
```

### 1301. Number of Paths with Max Score

```python
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        @cache
        def dfs(r, c):
            if r == c == n - 1:
                return [0, 1]
            res = [-inf, 0]
            for dr, dc in [[r + 1, c + 1], [r + 1, c], [r, c + 1]]:
                if 0 <= dr < n and 0 <= dc < n:
                    nxt = dfs(dr, dc)
                    if res[0] < nxt[0]:
                        res = nxt[:]
                    elif res[0] == nxt[0]:
                        res[1] = (res[1] + nxt[1]) % mod 
            res[0] += d[board[r][c]]
            return res 
        
        mod = 10 ** 9 + 7
        d = {str(i): i for i in range(10)}
        d['S'] = 0
        d['E'] = 0
        d['X'] = -inf 
        n = len(board)
        res = dfs(0, 0)
        return res if res[0] > -inf else [0, 0]
```

### 2435. Paths in Matrix Whose Sum Is Divisible by K

```python
class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        mod = 10 ** 9 + 7
        R, C = len(grid), len(grid[0])
        @cache
        def dfs(r, c, total):
            if r >= R or c >= C:
                return 0
            total = (total + grid[r][c]) % k
            if r == R - 1 and c == C - 1:
                return 1 if total % k == 0 else 0
            res = dfs(r + 1, c, total) + dfs(r, c + 1, total)
            return res % mod
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return res
```

### 174. Dungeon Game

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        @cache 
        def dfs(r, c, total):
            if r >= R or c >= C:
                return False 
            total += dungeon[r][c]
            if r == R - 1 and c == C - 1:
                return True if total > 0 else False 
            if total > 0:
                return dfs(r + 1, c, total) or dfs(r, c + 1, total)

        R, C = len(dungeon), len(dungeon[0])
        l, r, res = 1, 4 * 10 ** 7, 1
        while l <= r:
            m = l + (r - l) // 2 
            if dfs(0, 0, m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res 
```

### 329. Longest Increasing Path in a Matrix

```python 
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        @cache
        def dfs(r, c):
            res = 1
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                    res = max(res, dfs(row, col) + 1)
            return res 
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        res = 1
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
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                    res += dfs(row, col)
            return res 
            
        for r in range(R):
            for c in range(C):
                res += dfs(r, c)
        return res % mod 
```

### 329. Longest Increasing Path in a Matrix

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        @cache
        def dfs(r, c):
            res = 0
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                    res = max(res, dfs(row, col) + 1)
            return res 
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        res = 0
        for r in range(R):
            for c in range(C):
                res = max(res, dfs(r, c))
        return res + 1
```

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        R, C = len(matrix), len(matrix[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        indegree = [0] * R * C 

        for r in range(R):
            for c in range(C):
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[row * C + col] += 1

        res, q = 0, deque([i for i, d in enumerate(indegree) if d == 0])
        while q:
            for i in range(len(q)):
                node = q.popleft()
                r, c = node // C,  node % C
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and matrix[r][c] < matrix[row][col]:
                        indegree[row * C + col] -= 1
                        if indegree[row * C + col] == 0:
                            q.append(row * C + col)
            res += 1
        return res
```

### 2267. Check if There Is a Valid Parentheses String Path

```python
class Solution:
    def hasValidPath(self, grid: List[List[str]]) -> bool:
        @cache
        def dfs(r, c, open, close):
            if r >= R or c >= C or open < close:
                return False 
            if r == R - 1 and c == C - 1 and grid[r][c] == ')':
                return True if open == close + 1 else False 
            if grid[r][c] == '(':
                return dfs(r + 1, c, open + 1, close) or dfs(r, c + 1, open + 1, close)
            else:
                return dfs(r + 1, c, open, close + 1) or dfs(r, c + 1, open, close + 1)
        
        R, C = len(grid), len(grid[0])
        res = dfs(0, 0, 0, 0)
        dfs.cache_clear()
        return res
```


### 2510. Check if There is a Path With Equal Number of 0's And 1's

- same as 2267. Check if There Is a Valid Parentheses String Path

```python
class Solution:
    def isThereAPath(self, grid: List[List[int]]) -> bool:
        @cache
        def dfs(r, c, one, zero):
            if r >= R or c >= C or one > (R + C) / 2 or zero > (R + C) / 2:
                return False 
            if r == R - 1 and c == C - 1:
                if grid[r][c] == 1:
                    return True if one + 1 == zero else False 
                else:
                    return True if one == zero + 1 else False 
            if grid[r][c] == 1:
                return dfs(r + 1, c, one + 1, zero) or dfs(r, c + 1, one + 1, zero)
            else:
                return dfs(r + 1, c, one, zero + 1) or dfs(r, c + 1, one, zero + 1)

        R, C = len(grid), len(grid[0])
        res = dfs(0, 0, 0, 0)
        dfs.cache_clear()
        return res
```

### 1937. Maximum Number of Points with Cost

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        R, C = len(points), len(points[0])
        dp = deepcopy(points)
        for r in range(1, R):
            preMax, sufMax = -inf, -inf
            for c in range(C):
                preMax = max(preMax, dp[r - 1][c] + c)
                dp[r][c] = max(dp[r][c], points[r][c] - c + preMax)
            for c in range(C - 1, -1, -1):
                sufMax = max(sufMax, dp[r - 1][c] - c)
                dp[r][c] = max(dp[r][c], points[r][c] + c + sufMax)
        return max(dp[-1])
# points[r][c] = points[r][c] + max(points[r - 1][k] + k - c) k <= c
# points[r][c] = points[r][c] + max(points[r - 1][k] - k + c) k >= c

# points[r][c] = points[r][c] - c + max(points[r - 1][k] + k) k <= c
# points[r][c] = points[r][c] + c + max(points[r - 1][k] - k) k >= c
```

### 1463. Cherry Pickup II

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache 
        def dfs(r, c1, c2):
            if r >= R or c1 < 0 or c1 >= C or c2 < 0 or c2 >= C:
                return -inf 
            cur = grid[r][c1] if c1 == c2 else grid[r][c1] + grid[r][c2]
            if r == R - 1:
                return cur
            res = -inf 
            for dr1, dc1 in directions:
                row1, col1 = r + dr1, c1 + dc1 
                for dr2, dc2 in directions:
                    row2, col2 = r + dr2, c2 + dc2 
                    res = max(res, dfs(row1, col1, col2))
            return res + cur
        R, C = len(grid), len(grid[0])
        directions = [[1, 0], [1, -1], [1, 1]]
        return dfs(0, 0, C - 1)
```

### 741. Cherry Pickup

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache 
        def dfs(r1, c1, r2): # c2 == r1 + c1 - r2
            if any(item >= n for item in [r1, c1, r2, r1 + c1 - r2]) or grid[r1][c1] == -1 or grid[r2][r1 + c1 - r2] == -1:
                return -inf 
            cur = grid[r1][c1] if c1 == r1 + c1 - r2 else grid[r1][c1] + grid[r2][r1 + c1 - r2]
            if r1 == n - 1 and c1 == n - 1:
                return cur
            res = -inf 
            for dr1, dc1 in directions:
                row1, col1 = r1 + dr1, c1 + dc1 
                for dr2, dc2 in directions:
                    row2, col2 = r2 + dr2, r1 + c1 - r2 + dc2 
                    res = max(res, dfs(row1, col1, row2))
            return res + cur
        n = len(grid)
        directions = [[1, 0], [0, 1]]
        res = dfs(0, 0, 0)
        return res if res != -inf else 0
```

### 3363. Find the Maximum Number of Fruits Collected

```python
class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        @cache
        def dfs(i, j):
            if not (n - 1 - i <= j < n):
                return -inf
            if i == 0:
                return fruits[i][j]
            return max(dfs(i - 1, j - 1), dfs(i - 1, j), dfs(i - 1, j + 1)) + fruits[i][j]

        ans = sum(row[i] for i, row in enumerate(fruits))
        ans += dfs(n - 2, n - 1) 
        dfs.cache_clear()
        fruits = list(zip(*fruits))
        return ans + dfs(n - 2, n - 1)
```

### 3459. Length of Longest V-Shaped Diagonal Segment

```python
class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        R, C = len(grid), len(grid[0])
        @cache 
        def dfs(r, c, k, can_turn, target):
            r += directions[k][0]
            c += directions[k][1]
            if not (0 <= r < R and 0 <= c < C) or grid[r][c] != target:
                return 0
            res = dfs(r, c, k, can_turn, 2 - target)
            if can_turn:
                res = max(res, dfs(r, c, (k + 1) % 4, False, 2 - target))
            return res + 1

        res = 0
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if x == 1:
                    for k in range(4):
                        res = max(res, dfs(i, j, k, True, 2) + 1)
        return res  
```