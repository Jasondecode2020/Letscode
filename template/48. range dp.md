## range dp

- 3 loops of i, j, k
- dp[i][j] from 1 to i numbers choose j split and get the result
- dp[i][j] = max(dp[i][j], dp[k - 1][j - 1] + from j to i)

### 813. Largest Sum of Averages

```python
class Solution:
    def largestSumOfAverages(self, nums: List[int], m: int) -> float:
        n = len(nums)
        nums = list(accumulate(nums, initial = 0))
        dp = [[0] * (m + 1) for c in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                if j == 1:
                    dp[i][j] = nums[i] / i 
                else:
                    for k in range(2, i + 1):
                        dp[i][j] = max(dp[i][j], dp[k - 1][j - 1] + (nums[i] - nums[k - 1]) / (i - k + 1))
        return dp[n][m]
```

### 410. Split Array Largest Sum

```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        n = len(nums)
        nums = list(accumulate(nums, initial = 0))
        dp = [[inf] * (m + 1) for c in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                if j == 1:
                    dp[i][j] = nums[i]
                else:
                    for k in range(2, i + 1):
                        t = dp[k - 1][j - 1]
                        if t < nums[i] - nums[k - 1]:
                            t = nums[i] - nums[k - 1]
                        if t < dp[i][j]:
                            dp[i][j] = t
                        # dp[i][j] = min(dp[i][j], max(dp[k - 1][j - 1], (nums[i] - nums[k - 1])))
        return dp[n][m]
```