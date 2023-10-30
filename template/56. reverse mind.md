### 918. Maximum Sum Circular Subarray

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        maxNums, minNums = [nums[0]] * n, [nums[0]] * n
        for i in range(1, n):
            maxNums[i] = max(nums[i], maxNums[i - 1] + nums[i])
        for i in range(1, n):
            minNums[i] = min(nums[i], minNums[i - 1] + nums[i])
        if sum(nums) - min(minNums) == 0:
            return max(maxNums)
        return max(max(maxNums), sum(nums) - min(minNums))
```