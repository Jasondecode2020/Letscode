## template

### 2444. Count Subarrays With Fixed Bounds

```python
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        l, r1, r2, res = -1, -1, -1, 0
        for i in range(len(nums)):
            if nums[i] > maxK or nums[i] < minK: 
                l = i
            if nums[i] == maxK: 
                r1 = i
            if nums[i] == minK: 
                r2 = i
            res += max(0, min(r1, r2) - l)
        return res
```