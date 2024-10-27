### 2944. Minimum Number of Coins for Fruits

- house robber

```python
class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        @cache
        def f(i):
            if i >= n:
                return 0
            return min(f(j) for j in range(i + 1, i + i + 3)) + prices[i]
        n = len(prices)
        return f(0)
```

### 2140. Solving Questions With Brainpower

```python
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        @cache
        def dfs(i):
            if i >= n:
                return 0
            return max(dfs(i + 1), dfs(i + questions[i][1] + 1) + questions[i][0])

        n = len(questions)
        return dfs(0)
```

### 983. Minimum Cost For Tickets

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        last_day = days[-1]
        days = set(days)
        @cache
        def dfs(i):
            if i <= 0:
                return 0 
            if i not in days:
                return dfs(i - 1)
            return min(dfs(i - 1) + costs[0], dfs(i - 7) + costs[1], dfs(i - 30) + costs[2])
        return dfs(last_day)
```

### 651. 4 Keys Keyboard

```python
class Solution:
    def maxA(self, n: int) -> int:
        dp = [i for i in range(n + 1)]
        for i in range(4, n + 1):
            for j in range(i - 2):
                dp[i] = max(dp[i], dp[j] * (i - j - 2 + 1))
        return dp[-1]
```

### 650. 2 Keys Keyboard

```python
class Solution:
    def minSteps(self, n: int) -> int:
        dp = [inf] * (n + 1)
        dp[1] = 0
        for i in range(2, n + 1):
            for j in range(2, i + 1):
                if i % j == 0:
                    k = i // j
                    dp[i] = min(dp[i], dp[k] + 1 + j - 1)
        return dp[-1]
```

### 2369. Check if There is a Valid Partition For The Array

```python
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        @cache
        def dfs(i):
            if i == n:
                return True  
            first, second = False, False          
            if i + 1 < n and nums[i] == nums[i + 1]:
                first = dfs(i + 2)
            if i + 2 < n and (nums[i] == nums[i + 1] == nums[i + 2] or (nums[i] + 1 == nums[i + 1] and nums[i + 1] + 1 == nums[i + 2])):
                second = dfs(i + 3) 
            return first or second 
        return dfs(0)
```

### 139. Word Break

- prefix idea: dp[i] means if dp[:i] is combined by the wordDict or not

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n, wordDict = len(s), set(wordDict)
        dp = [True] + [False] * n
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]
```


### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res, lowest = 0, prices[0]
        for p in prices:
            lowest = min(lowest, p)
            res = max(res, p - lowest)
        return res
```

### 1025. Divisor Game

```python
class Solution:
    def divisorGame(self, n: int) -> bool:
        @cache
        def dfs(n):
            if n == 1:
                return False

            for i in range(1, n):
                if n % i == 0:
                    if not dfs(n - i):
                        return True
            return False
        return dfs(n)
```

### 118. Pascal's Triangle

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for i in range(2, numRows + 1):
            lastRow = res[-1]
            lastRow = [0] + lastRow + [0]
            lastRow = [lastRow[i] + lastRow[i + 1] for i in range(len(lastRow) - 1)]
            res.append(lastRow)
        return res
```

### 119. Pascal's Triangle II

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [[1]]
        for i in range(1, rowIndex + 1):
            lastRow = res[-1]
            lastRow = [0] + lastRow + [0]
            lastRow = [lastRow[i] + lastRow[i + 1] for i in range(len(lastRow) - 1)]
            res.append(lastRow)
        return res[-1]
```

### 338. Counting Bits

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [[1]]
        for i in range(1, rowIndex + 1):
            lastRow = res[-1]
            lastRow = [0] + lastRow + [0]
            lastRow = [lastRow[i] + lastRow[i + 1] for i in range(len(lastRow) - 1)]
            res.append(lastRow)
        return res[-1]
```

### 2826. Sorting Three Groups

```python
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * 3 for r in range(n + 1)]
        for i in range(1, n + 1):
            if nums[i - 1] == 1:
                dp[i][0] = dp[i - 1][0]
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + 1
                dp[i][2] = min(dp[i - 1][0], dp[i - 1][1], dp[i - 1][2]) + 1
            elif nums[i - 1] == 2:
                dp[i][0] = dp[i - 1][0] + 1
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1])
                dp[i][2] = min(dp[i - 1][0], dp[i - 1][1], dp[i - 1][2]) + 1
            else:
                dp[i][0] = dp[i - 1][0] + 1
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + 1
                dp[i][2] = min(dp[i - 1][0], dp[i - 1][1], dp[i - 1][2])
        return min(dp[-1])
```

### 1770. Maximum Score from Performing Multiplication Operations

```python
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        m = len(multipliers)
        @cache
        def dfs(l, r, idx):
            if idx == m:
                return 0
            res = -inf
            res = max(res, nums[l] * multipliers[idx] + dfs(l + 1, r, idx + 1))
            res = max(res, nums[r] * multipliers[idx] + dfs(l, r - 1, idx + 1))
            return res
        return dfs(0, len(nums) - 1, 0)
```

### 1473. Paint House III

```python
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        @cache
        def dfs(i, last, t): # index of house, last colored, target
            if i == m:
                return inf if t != target else 0
            if t > target:
                return inf 
            if houses[i] == 0:
                res = inf
                for color in range(1, n + 1):
                    res = min(res, dfs(i + 1, color, t + (last != color)) + cost[i][color - 1])
                return res 
            else:
                return dfs(i + 1, houses[i], t + (last != houses[i]))
        res = dfs(0, -1, 0)
        return res if res != inf else -1
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

### 1463. Cherry Pickup II

```python
```

### 1335. Minimum Difficulty of a Job Schedule

```python
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        n = len(jobDifficulty)
        if n < d:
            return -1
        suf = [jobDifficulty[-1]] * n 
        for i in range(n - 2, -1, -1):
            suf[i] = max(jobDifficulty[i], suf[i + 1])
        @cache
        def dfs(i, day):
            if day == d:
                return suf[i] if i < n else -inf 
            if day > d:
                return -inf 
            res, mx = inf, 0
            for j in range(i + 1, n):
                mx = max(mx, jobDifficulty[j - 1])
                res = min(res, dfs(j, day + 1) + mx)
            return res
        res = dfs(0, 1)
        return res if res != -inf else -1
```

### 1575. Count All Possible Routes

```python
class Solution:
    def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
        @cache
        def dfs(i, f):
            if i == finish:
                if f <= fuel:
                    ans = 0
                    for j in range(n):
                        if j != i:
                            ans += dfs(j, f + abs(locations[i] - locations[j]))
                    return 1 + ans
                return 0
            if f > fuel:
                return 0
            res = 0
            for j in range(n):
                if j != i:
                    res += dfs(j, f + abs(locations[i] - locations[j]))
            return res
        n = len(locations)
        mod = 10 ** 9 + 7
        return dfs(start, 0) % mod
```

### 1639. Number of Ways to Form a Target String Given a Dictionary

```python
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        n = len(words[0])
        m = len(target)
        mod = 10 ** 9 + 7
        dp = [[0] * 26 for i in range(n)]
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                dp[j][ord(words[i][j]) - ord('a')] += 1
        @cache
        def dfs(i, k):
            if k == m:
                return 1
            if i == n or k > m:
                return 0
            res = 0
            res += dp[i][ord(target[k]) - ord('a')] * dfs(i + 1, k + 1)
            res += dfs(i + 1, k)
            return res 
        return dfs(0, 0) % mod
```

### 576. Out of Boundary Paths

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        @cache
        def dfs(r, c, move):
            if move > maxMove:
                return 0
            if r < 0 or r == R or c < 0 or c == C:
                return 1
            res = 0
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                res += dfs(row, col, move + 1)
            return res % mod 

        mod = 10 ** 9 + 7
        R, C = m, n 
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        return dfs(startRow, startColumn, 0)
```

### 2318. Number of Distinct Roll Sequences

```python
class Solution:
    def distinctSequences(self, n: int) -> int:
        @cache
        def dfs(i, j, prev_j):
            if i == n:
                return 1
            res = 0
            for k in range(1, 7):
                if gcd(j, k) == 1 and k != j and k != prev_j:
                    res += dfs(i + 1, k, j)
            return res % mod 
        
        mod = 10 ** 9 + 7
        return dfs(0, -1, -1) % mod 
```

### 1223. Dice Roll Simulation

```python
class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        @cache
        def dfs(i, last, cnt):
            if i == n:
                return 1
            res = 0
            for j in range(1, 7):
                if j != last:
                    res += dfs(i + 1, j, 1)
                elif cnt < rollMax[j - 1]:
                    res += dfs(i + 1, j, cnt + 1)
            return res % mod
        
        mod = 10 ** 9 + 7
        res = dfs(0, -1, 0)
        return res 
```