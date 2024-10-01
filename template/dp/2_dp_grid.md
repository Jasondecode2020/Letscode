
## 2 Grid (18)

### 2.1 Basics (8)

* [62. Unique Paths](#62-unique-paths)
* [63. Unique Paths II](#63-unique-paths-ii)
* [64. Minimum Path Sum](#64-minimum-path-sum)
* [120. Triangle](#120-triangle)
* [931. Minimum Falling Path Sum](#931-minimum-falling-path-sum)
* [2684. Maximum Number of Moves in a Grid](#2684-maximum-number-of-moves-in-a-grid)
* [1289. Minimum Falling Path Sum II](#1289-minimum-falling-path-sum-ii)
* [2304. Minimum Path Cost in a Grid](#2304-minimum-path-cost-in-a-grid)

### 2.2 Advanced (10)

* [1594. Maximum Non Negative Product in a Matrix](#62-unique-paths)
* [2435. Paths in Matrix Whose Sum Is Divisible by K](#2435-paths-in-matrix-whose-sum-is-divisible-by-k)
* [174. Dungeon Game](#174-dungeon-game)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2267. Check if There Is a Valid Parentheses String Path](#2267-check-if-there-is-a-valid-parentheses-string-path)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2510. Check if There is a Path With Equal Number of 0's And 1's](#2510-check-if-there-is-a-path-with-equal-number-of-0s-and-1s)
* [1463. Cherry Pickup II](#1463-cherry-pickup-ii)
* [741. Cherry Pickup](#741-cherry-pickup)
* [1937. Maximum Number of Points with Cost](#1937-maximum-number-of-points-with-cost)

## 3 Knapsack (18)

### 3.1 0-1 (8)

* [62. Unique Paths](#62-unique-paths)
* [63. Unique Paths II](#63-unique-paths-ii)
* [64. Minimum Path Sum](#64-minimum-path-sum)
* [120. Triangle](#120-triangle)
* [931. Minimum Falling Path Sum](#931-minimum-falling-path-sum)
* [2684. Maximum Number of Moves in a Grid](#2684-maximum-number-of-moves-in-a-grid)
* [1289. Minimum Falling Path Sum II](#1289-minimum-falling-path-sum-ii)
* [2304. Minimum Path Cost in a Grid](#2304-minimum-path-cost-in-a-grid)

### 3.2 unbounded (10)

* [1594. Maximum Non Negative Product in a Matrix](#62-unique-paths)
* [2435. Paths in Matrix Whose Sum Is Divisible by K](#2435-paths-in-matrix-whose-sum-is-divisible-by-k)
* [174. Dungeon Game](#174-dungeon-game)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2267. Check if There Is a Valid Parentheses String Path](#2267-check-if-there-is-a-valid-parentheses-string-path)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2510. Check if There is a Path With Equal Number of 0's And 1's](#2510-check-if-there-is-a-path-with-equal-number-of-0s-and-1s)
* [1463. Cherry Pickup II](#1463-cherry-pickup-ii)
* [741. Cherry Pickup](#741-cherry-pickup)
* [1937. Maximum Number of Points with Cost](#1937-maximum-number-of-points-with-cost)

### 3.3 mutiple (1)

* [1594. Maximum Non Negative Product in a Matrix](#62-unique-paths)

### 3.2 group (3)

* [1155. Number of Dice Rolls With Target Sum](#1155-number-of-dice-rolls-with-target-sum)
* [1981. Minimize the Difference Between Target and Chosen Elements](#2435-paths-in-matrix-whose-sum-is-divisible-by-k)
* [2218. Maximum Value of K Coins From Piles](#174-dungeon-game)

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
        R, C = len(grid), len(grid[0])
        dp = [[0] * C for r in range(R)]
        for i, n in enumerate(grid[0]):
            dp[0][i] = n
        for r in range(1, R):
            for c in range(C):
                dp[r][c] = grid[r][c] + min(dp[r - 1][: c] + dp[r - 1][c + 1: ])
        return min(dp[-1])
```

### 2304. Minimum Path Cost in a Grid

```python
class Solution:
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dp = deepcopy(grid)
        for r in range(1, R):
            for c in range(C):
                dp[r][c] += min(dp[r - 1][j] + moveCost[grid[r - 1][j]][c] for j in range(C))
        return min(dp[-1]) 
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
        R, C = len(dungeon),len(dungeon[0])
        @cache
        def check(r, c, threshold):
            if r >= R or c >= C:
                return False
            total = threshold + dungeon[r][c]
            if r == R - 1 and c == C - 1:
                return True if total > 0 else False
            if total > 0:
                return check(r + 1, c, total) or check(r, c + 1, total)

        l, r, res = 1, 10 ** 9, 1
        while l <= r:
            mid = (l+r)//2
            if check(0, 0, mid):
                res = mid
                r = mid - 1
            else:
                l = mid + 1
        del check
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

### 329. Longest Increasing Path in a Matrix

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
            if r == R - 1 and c == C - 1:
                if grid[r][c] == '(':
                    return True if open + 1 == close else False 
                else:
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

### 1463. Cherry Pickup II

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(r1, c1, c2):
            if c1 < 0 or c1 >= C or c2 < 0 or c2 >= C:
                return -inf 
            if r1 == R - 1:
                return grid[r1][c1] if c1 == c2 else grid[r1][c1] + grid[r1][c2]
            cur = grid[r1][c1] if c1 == c2 else grid[r1][c1] + grid[r1][c2]
            nxt = -inf 
            for dr1, dc1 in directions:
                for dr2, dc2 in directions:
                    row1, col1 = r1 + dr1, c1 + dc1
                    row2, col2 = r1 + dr2, c2 + dc2
                    nxt = max(nxt, dfs(row1, col1, col2))
            return cur + nxt

        R, C = len(grid), len(grid[0])
        directions = [[1, -1], [1, 0], [1, 1]]
        return dfs(0, 0, C - 1)
```

### 741. Cherry Pickup

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @cache
        def dfs(r1, c1, r2): # r1 + c1 - r2 == c2
            if any([n >= N for n in [r1, c1, r2, r1 + c1 - r2]]) or grid[r1][c1] == -1 or grid[r2][r1 + c1 - r2] == -1:
                return -inf 
            if r1 == c1 == N - 1:
                return grid[r1][c1]

            cur = grid[r1][c1] if r1 == r2 and c1 == r1 + c1 - r2 else grid[r1][c1] + grid[r2][r1 + c1 - r2]
            nxt = -inf 
            for dr1, dc1 in directions:
                for dr2, dc2 in directions:
                    row1, col1 = r1 + dr1, c1 + dc1 
                    row2, col2 = r2 + dr2, r1 + c1 - r2 + dc2 
                    nxt = max(nxt, dfs(row1, col1, row2))
            return cur + nxt

        N = len(grid)
        directions = [[0, 1], [1, 0]]
        res = dfs(0, 0, 0)
        return res if res != -inf else 0
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
```

### 2915. Length of the Longest Subsequence That Sums to Target

```python
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        nums.sort()
        N = len(nums)
        @cache
        def f(t, i):
            if t > target:
                return -inf 
            if i == N:
                return 0 if t == target else -inf 
            return max(f(t, i + 1), f(t + nums[i], i + 1) + 1)
        res = f(0, 0)
        f.cache_clear()
        return res if res != -inf else -1
```

### 416. Partition Equal Subset Sum

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 == 1:
            return False
        half = sum(nums) // 2
        @cache
        def f(t, i):
            if i == len(nums):
                return True if t == half else False
            return f(t, i + 1) or f(t + nums[i], i + 1)
        return f(0, 0)
```

### 494. Target Sum

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        @cache
        def f(t, i):
            if i == len(nums):
                return 1 if t == target else 0
            return f(t - nums[i], i + 1) + f(t + nums[i], i + 1)
        return f(0, 0)
```

### 2787. Ways to Express an Integer as Sum of Powers

```python
mod = 10 ** 9 + 7
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        nums = [i for i in range(n, 0, -1)]
        N = len(nums)
        @cache
        def f(t, i):
            if t > n:
                return 0
            if i == N:
                return 1 if t == n else 0
            return (f(t, i + 1) + f(t + nums[i] ** x, i + 1)) % mod
        res = f(0, 0)
        return res
```

### 474. Ones and Zeroes

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        @cache
        def f(a, b, i):
            if i == len(strs):
                return 0 if (a <= m and b <= n) else -inf
            if a > m or b > n:
                return -inf
            ones, zeros = strs[i].count('1'), strs[i].count('0')
            return max(f(a, b, i + 1), f(a + zeros, b + ones, i + 1) + 1)
        res = f(0, 0, 0)
        return res if res != -inf else 0
```

### 1049. Last Stone Weight II

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        mx = max(stones)
        @cache
        def f(t, i):
            if i == len(stones):
                return abs(t)
            if t > mx:  # optimize
                return inf
            return min(f(t - stones[i], i + 1), f(t + stones[i], i + 1))

        return f(0, 0)
```

1774. 最接近目标价格的甜点成本
879. 盈利计划 2204
3082. 求出所有子序列的能量和 2242
956. 最高的广告牌 2381
2518. 好分区的数目 2415
2742. 给墙壁刷油漆 2425
LCP 47. 入场安检
2291. 最大股票收益（会员题）
2431. 最大限度地提高购买水果的口味（会员题）

# ###################################################################################


### 279. Perfect Squares

```python
square = [i * i for i in range(100, 0, -1)]
class Solution:
    def numSquares(self, n: int) -> int:
        @cache
        def f(t, i):
            if t > n:
                return inf 
            if i == len(square):
                return 0 if t == n else inf
            return min(f(t, i + 1), f(t + square[i], i) + 1) 
        res = f(0, 0)
        f.cache_clear()
        return res
```

### 322. Coin Change

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse = True)
        @cache
        def f(t, i):
            if t > amount:
                return inf 
            if i == len(coins):
                return 0 if t == amount else inf 
            return min(f(t, i + 1), f(t + coins[i], i) + 1) 
        res = f(0, 0)
        return res if res != inf else -1
```

### 518. Coin Change II

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        coins.sort(reverse = True)
        N = len(coins)
        @cache
        def f(t, i):
            if t > amount:
                return 0
            if i == N:
                return 1 if t == amount else 0
            return f(t, i + 1) + f(t + coins[i], i)
        res = f(0, 0)
        return res if res != inf else -1
```

### 377. Combination Sum IV (unbounded knapsack with sequence)

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @cache
        def f(t):
            if t > target:
                return 0
            if t == target:
                return 1
            return sum(f(t + n) for n in nums)
        return f(0)
```

### 2585. Number of Ways to Earn Points

```python
mod = 10 ** 9 + 7
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        n = len(types)
        @cache
        def f(i, t):
            if i < 0:
                return 1 if t == 0 else 0
            count, mark = types[i]
            res = 0
            for k in range(min(count, t // mark) + 1):
                res += f(i - 1, t - k * mark)
            return res
        return f(n - 1, target) % mod
```

### 1449. Form Largest Integer With Digits That Add up to Target

###########################################################################################

### 2585. Number of Ways to Earn Points

```python
mod = 10 ** 9 + 7
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        n = len(types)
        @cache
        def f(i, t):
            if i < 0:
                return 1 if t == 0 else 0
            count, mark = types[i]
            res = 0
            for k in range(min(count, t // mark) + 1):
                res += f(i - 1, t - k * mark)
            return res
        return f(n - 1, target) % mod
```

#####################################################################################################

### 1155. Number of Dice Rolls With Target Sum

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, t):
            if t > target:
                return 0
            if i == n:
                return 1 if t == target else 0
            res = 0
            for j in range(1, k + 1):
                res += dfs(i + 1, t + j)
            return res % mod
        return dfs(0, 0) % mod
```

### 1981. Minimize the Difference Between Target and Chosen Elements

### 2218. Maximum Value of K Coins From Piles

## 5 State Machine (18)

* [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
* [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
* [123. Best Time to Buy and Sell Stock III](#123-best-time-to-buy-and-sell-stock-iii)
* [188. Best Time to Buy and Sell Stock IV](#188-best-time-to-buy-and-sell-stock-iv)
* [309. Best Time to Buy and Sell Stock with Cooldown](#309-best-time-to-buy-and-sell-stock-with-cooldown)
* [714. Best Time to Buy and Sell Stock with Transaction Fee](#714-best-time-to-buy-and-sell-stock-with-transaction-fee)

### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest, profit = prices[0], 0
        for price in prices:
            lowest = min(lowest, price)
            profit = max(profit, price - lowest)
        return profit
```

### 122. Best Time to Buy and Sell Stock II

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # [7,1,5,3,6,4]
        res = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                res += prices[i] - prices[i - 1]
        return res
```

### 123. Best Time to Buy and Sell Stock III

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding, count):
            if i >= n or count == 2:
                return 0
            cooldown = dfs(i + 1, holding, count)
            if holding:
                sell = dfs(i + 1, False, count + 1) + prices[i]
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True, count) - prices[i]
                res = max(cooldown, buy)
            return res 
        return dfs(0, False, 0)
```

### 188. Best Time to Buy and Sell Stock IV

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding, count):
            if i >= n or count == k:
                return 0
            cooldown = dfs(i + 1, holding, count)
            if holding:
                sell = dfs(i + 1, False, count + 1) + prices[i]
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True, count) - prices[i]
                res = max(cooldown, buy)
            return res 
        return dfs(0, False, 0)
```

### 309. Best Time to Buy and Sell Stock with Cooldown

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding):
            if i >= n:
                return 0
            cooldown = dfs(i + 1, holding)
            if holding:
                sell = dfs(i + 2, False) + prices[i]
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True) - prices[i]
                res = max(cooldown, buy)
            return res
        return dfs(0, False)
```

### 714. Best Time to Buy and Sell Stock with Transaction Fee

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding):
            if i >= n:
                return 0
            cooldown = dfs(i + 1, holding)
            if holding:
                sell = dfs(i + 1, False) + prices[i] - fee 
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True) - prices[i]
                res = max(cooldown, buy)
            return res 
        return dfs(0, False)
```

### 1493. Longest Subarray of 1's After Deleting One Element

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        l, res = 0, 0
        zero = 0
        for r, n in enumerate(nums):
            if n == 0:
                zero += 1
            while zero > 1:
                if nums[l] == 0:
                    zero -= 1
                l += 1
            res = max(res, r - l)
        return res 
```

### 1395. Count Number of Teams

```python
class Solution:
    def numTeams(self, rating: List[int]) -> int:
        res = 0
        n = len(rating)
        for i in range(1, n - 1):
            left = sum(rating[j] < rating[i] for j in range(i))
            right = sum(rating[j] > rating[i] for j in range(i + 1, n))
            res += left * right
            left = sum(rating[j] > rating[i] for j in range(i))
            right = sum(rating[j] < rating[i] for j in range(i + 1, n))
            res += left * right
        return res
```