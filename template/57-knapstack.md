## equal to target

### 2915. Length of the Longest Subsequence That Sums to Target

- 0-1 knapsack

```python
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        f = [0] + [-inf] * target # first number to be 0
        for x in nums:
            for i in range(target, x - 1, -1): # start from target to x, can't choose duplicate
                f[i] = max(f[i], f[i - x] + 1) # try to find max
        return f[-1] if f[-1] > 0 else -1
```

### 377. Combination Sum IV

- backtrack not possible

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        self.res, n = 0, len(nums)
        def backtrack(idx, cur):
            if cur == target:
                self.res += 1
                return 
            if cur > target:
                return
            for i in range(0, n):
                backtrack(i, cur + nums[i])
        backtrack(0, 0)
        return self.res
```

- unbounded knapsack

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1] + [0] * target
        for i in range(1, target + 1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i - num]
        return dp[-1]
```