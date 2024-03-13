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