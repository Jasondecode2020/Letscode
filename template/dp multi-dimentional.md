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