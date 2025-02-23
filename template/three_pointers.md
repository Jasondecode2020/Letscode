## template

* [2367. Number of Arithmetic Triplets](#2367-number-of-arithmetic-triplets)
* [2563. Count the Number of Fair Pairs](#2563-count-the-number-of-fair-pairs)
* [795. Number of Subarrays with Bounded Maximum](#795-number-of-subarrays-with-bounded-maximum)
* [2444. Count Subarrays With Fixed Bounds](#2444-count-subarrays-with-fixed-bounds)
* [1213. Intersection of Three Sorted Arrays](#1213-intersection-of-three-sorted-arrays)

- TODO:
* [3347. Maximum Frequency of an Element After Performing Operations II](#3347-maximum-frequency-of-an-element-after-performing-operations-ii)

### 2367. Number of Arithmetic Triplets

- O(n^3)

```python 
class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        n, res = len(nums), 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if 2 * nums[j] == nums[i] + nums[k] and nums[j] - nums[i] == diff:
                        res += 1
        return res
```
- O(n^2)

```python 
class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        n, res = len(nums), 0
        s = set(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if 2 * nums[j] - nums[i] in s and nums[j] - nums[i] == diff:
                    res += 1
        return res
```

- O(n)

```python
class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        s = set(nums)
        res = 0
        for i, n in enumerate(nums):
            if n - diff in s and n + diff in s:
                res += 1
        return res
```

- O(n), S(1)

```python
class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        res, i, j = 0, 0, 1
        for x in nums:
            while nums[j] + diff < x:
                j += 1
            if nums[j] + diff != x:
                continue 
            while nums[i] + diff < nums[j]:
                i += 1
            if nums[i] + diff == nums[j]:
                res += 1
        return res 
```

### 2563. Count the Number of Fair Pairs

```python 
class Solution:
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        nums.sort()
        res = 0
        for i, n in enumerate(nums):
            l, r = bisect_left(nums, lower - n, 0, i), bisect_right(nums, upper - n, 0, i)
            res += r - l 
        return res 
```

### 795. Number of Subarrays with Bounded Maximum

```python 
class Solution:
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        res = cnt = 0
        pre = -1
        for i, n in enumerate(nums):
            if left <= n <= right:
                cnt = i - pre
            elif n > right:
                pre = i 
                cnt = 0 
            res += cnt 
        return res 
```

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

### 1213. Intersection of Three Sorted Arrays

```python 
class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        res = []
        s1, s2, s3 = set(arr1), set(arr2), set(arr3)
        for i in range(1, 2001):
            if i in s1 and i in s2 and i in s3:
                res.append(i)
        return res
```

```python 
class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        res = []
        m, n, k = len(arr1), len(arr2), len(arr3)
        i = j = p = 0
        while i < m and j < n and p < k:
            upper = max(arr1[i], arr2[j], arr3[p])
            if arr1[i] != upper or arr2[j] != upper or arr3[p] != upper:
                if arr1[i] != upper:
                    i += 1
                if arr2[j] != upper:
                    j += 1
                if arr3[p] != upper:
                    p += 1
            else:
                res.append(upper)
                i += 1
                j += 1
                p += 1
        return res
```

### 3347. Maximum Frequency of an Element After Performing Operations II
