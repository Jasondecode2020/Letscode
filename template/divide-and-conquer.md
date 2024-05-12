## d&c

### 395. Longest Substring with At Least K Repeating Characters

```python
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) < k:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))
        return len(s)
```

### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        def dfs(l, r):
            if l == r:
                return nums[l]
            m = l + (r - l) // 2
            left, res = -inf, 0
            for i in range(m, l - 1, -1):
                res += nums[i]
                left = max(left, res)
            right, res = -inf, 0
            for i in range(m + 1, r + 1):
                res += nums[i]
                right = max(right, res)
            return max(dfs(l, m), dfs(m + 1, r), left + right)
        return dfs(0, len(nums) - 1)
```