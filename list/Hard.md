## Note: 131, 114, 394, 155

### 41. First Missing Positive

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        # set inf
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = inf 
        # set negative
        for i in range(n):
            if abs(nums[i]) != inf:
                v = abs(nums[i])
                nums[v - 1] = -abs(nums[v - 1])
        # find
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1
```