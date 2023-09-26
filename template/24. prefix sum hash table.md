## base case

### 724. Find Pivot Index

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        nums = list(accumulate(nums, initial = 0))
        for i, n in enumerate(nums):
            if i > 0 and nums[i - 1] == nums[-1] - nums[i]:
                return i - 1
        return -1
```

### 1480. Running Sum of 1d Array

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        return list(accumulate(nums))
```

## sliding window

### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        nums = [0] + nums
        l, r, res = 0, 1, inf
        while r < len(nums):
            if nums[r] - nums[l] >= target:
                res = min(res, r - l)
                l += 1
            else:
                r += 1
        return res if res != inf else 0
```

### 1004. Max Consecutive Ones III

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]

        nums = [0] + nums
        l, res = 0, 0
        for r in range(1, len(nums)):
            if nums[r] - nums[l] + k >= r - l:
                res = max(res, r - l)
            else:
                l += 1
        return res
```

### 525. Contiguous Array

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        res, count, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            if n:
                count += 1
            else:
                count -= 1
            if count in d:
                res = max(res, i - d[count])
            else:
                d[count] = i
        return res
```

## binary search

### 209. Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        
        nums, res = [0] + nums, inf
        for l, n in enumerate(nums):
            r = bisect_left(nums, n + target)
            if r < len(nums):
                res = min(res, r - l)
        return res if res != inf else 0
```

## subarray

### 325. Maximum Size Subarray Sum Equals k

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: -1}
        for i, n in enumerate(nums):
            presum += n
            if presum - k in d:
                res = max(res, i - d[presum - k])
            if presum not in d:
                d[presum] = i
        return res
```

### 560. Subarray Sum Equals K

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: 1} # [1, 2, 3]
        for n in nums:
            presum += n
            if presum - k in d:
                res += d[presum - k]
            d[presum] = d.get(presum, 0) + 1
        return res
```

### 974. Subarray Sums Divisible by K

```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: 1}
        for n in nums:
            presum += n 
            if presum % k in d:
                res += d[presum % k]
            d[presum % k] = d.get(presum % k, 0) + 1
        return res
```

### 523. Continuous Subarray Sum

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        presum, d = 0, {0: -1}
        for i, n in enumerate(nums):
            presum += n 
            if presum % k in d and i - d[presum % k] >= 2:
                return True
            if presum % k not in d:
                d[presum % k] = i
        return False
```

### 1074. Number of Submatrices That Sum to Target

```python
class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        R, C = len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]

        res = 0
        # turn each column as 1d array, and find all combinations
        for i in range(C):
            for j in range(i, C):
                # solve 1d subarray problem
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, d = 0, {0: 1}
                for n in nums:
                    presum += n
                    if presum - target in d:
                        res += d[presum - target]
                    d[presum] = d.get(presum, 0) + 1
        return res
```

### 363. Max Sum of Rectangle No Larger Than K

```python
from sortedcontainers import SortedList
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        res, R, C = -inf, len(matrix), len(matrix[0])
        for row in matrix:
            for c in range(C - 1):
                row[c + 1] += row[c]
        for i in range(C):
            for j in range(i, C):
                nums = [matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0) for k in range(R)]
                presum, sortedList = 0, SortedList([0])
                for n in nums:
                    presum += n
                    idx = sortedList.bisect_left(presum - k)
                    if idx < len(sortedList):
                        res = max(res, presum - sortedList[idx])
                    sortedList.add(presum)
        return res
```