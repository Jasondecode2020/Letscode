### 1818. Minimum Absolute Sum Difference

```python
class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        res, mx = 0, 0
        nums = [-inf] + nums1 + [inf]
        nums.sort()
        mod = 10 ** 9 + 7
        for i, (a, b) in enumerate(zip(nums1, nums2)):
            res += abs(a - b)
            j = bisect_left(nums, b)
            mn = min(b - nums[j - 1], nums[j] - b)
            mx = max(mx, abs(a - b) - mn)
        return (res - mx) % mod
```

### 1649. Create Sorted Array through Instructions

```python
from sortedcontainers import SortedList
class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        mod = 10 ** 9 + 7
        sl, res = SortedList(), 0
        c = Counter()
        for i in instructions:
            sl.add(i)
            if len(sl) > 2:
                j = sl.bisect_left(i)
                mn = min(j, len(sl) - j - 1 - c[i])
                res += mn 
            c[i] += 1
        return res % mod
```