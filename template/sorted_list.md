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

## template: sorted list

### 315. Count of Smaller Numbers After Self

```python
from sortedcontainers import SortedList
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        sl = SortedList()
        res = []
        for n in nums[::-1]:
            i = SortedList.bisect_left(sl, n)
            sl.add(n)
            res.append(i)
        return res[::-1]
```

### 493. Reverse Pairs

```python
from sortedcontainers import SortedList
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        res = 0
        sl = SortedList()
        for n in nums[::-1]:
            i = SortedList.bisect_left(sl, n)
            res += i
            sl.add(2 * n)
        return res
```

### 327. Count of Range Sum

```python
from sortedcontainers import SortedList
class Solution: # faster
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        sl, res = SortedList(), 0
        for n in accumulate(nums, initial = 0):
            res += sl.bisect_right(n - lower) - sl.bisect_left(n - upper)
            sl.add(n)
        return res
```

```python
class Solution: # mush slower
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        sl, res = [], 0
        for n in accumulate(nums, initial = 0):
            res += bisect_right(sl, n - lower) - bisect_left(sl, n - upper)
            bisect.insort(sl, n)
        return res
```

### 295. Find Median from Data Stream

```python
from sortedcontainers import SortedList
class MedianFinder:

    def __init__(self):
        self.sl = SortedList()

    def addNum(self, num: int) -> None:
        self.sl.add(num)

    def findMedian(self) -> float:
        n = len(self.sl)
        if n % 2 == 1:
            return self.sl[n // 2]
        return (self.sl[n // 2 - 1] + self.sl[n // 2]) / 2
```

### 658. Find K Closest Elements

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l, left, right, total = 0, -1, -1, 0
        ans = inf
        for r, n in enumerate(arr):
            total += abs(n - x)
            if r - l + 1 == k:
                if total < ans:
                    ans = total
                    left, right = l, r
                total -= abs(arr[l] - x)
                l += 1
        return arr[left: right + 1]
```

### 1358. Number of Substrings Containing All Three Characters

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        l, res = 0, 0
        d = Counter(['a', 'b', 'c'])
        d_s = Counter()
        for r, c in enumerate(s):
            d_s[c] += 1
            while d_s >= d:
                res += n - r
                d_s[s[l]] -= 1
                l += 1
        return res
```

### 683. K Empty Slots

```python
from sortedcontainers import SortedList
class Solution:
    def kEmptySlots(self, bulbs: List[int], k: int) -> int:
        sl, n = SortedList([-inf, inf]), len(bulbs)
        for idx, position in enumerate(bulbs):
            sl.add(position)
            i = sl.bisect_left(position)
            if sl[i] - sl[i - 1] == k + 1 or sl[i + 1] - sl[i] == k + 1:
                return idx + 1
        return -1
```

### 2426. Number of Pairs Satisfying Inequality

```python
from sortedcontainers import SortedList
class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], diff: int) -> int:
        # [3,2,5], [2,2,1] => [(3, 2), (2, 2), (5, 1)]
        # a, c, b, d => a - c <= b - d + diff => a - b <= c - d + diff
        res = 0
        sl = SortedList()
        for a, b in zip(nums1, nums2):
            c = a - b 
            res += sl.bisect_right(c + diff)
            sl.add(c)
        return res 
```