## template 1: O(n^2)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1] * n 
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)
```

## template 2: O(nlog(n))

```python
def LIS(arr): # more than or equal
    LIS = []
    for n in arr:
        i = bisect_right(LIS, n)
        if i == len(LIS):
            LIS.append(n)
        else:
            LIS[i] = n
    return len(LIS)   
```

```python
def LIS(arr): # strictly increasing
    LIS = []
    for n in arr:
        i = bisect_left(LIS, n)
        if i == len(LIS):
            LIS.append(n)
        else:
            LIS[i] = n
    return len(LIS)   
```

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        def LIS(nums): # space O(1)
            end = 0
            for n in nums:
                i = bisect_left(nums, n, 0, end)
                if i == end:
                    nums[end] = n
                    end += 1
                else:
                    nums[i] = n
            return end
        return LIS(nums) >= 3
```

### question list

* 300. Longest Increasing Subsequence
* 334. Increasing Triplet Subsequence
* 646. Maximum Length of Pair Chain
* 1626. Best Team With No Conflicts
* 354. Russian Doll Envelopes
* 1691. Maximum Height by Stacking Cuboids 

### 300. Longest Increasing Subsequence

- binary search
- O(nlog(n))

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def LIS(nums):
            LIS = []
            for n in nums:
                i = bisect_left(LIS, n)
                if i == len(LIS):
                    LIS.append(n)
                else:
                    LIS[i] = n 
            return len(LIS)
        return LIS(nums)
```

- O(n^2)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1] * n 
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)
```


### 334. Increasing Triplet Subsequence

- binary search 

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        def LIS(nums):
            LIS = []
            for n in nums:
                i = bisect_left(LIS, n)
                if i == len(LIS):
                    LIS.append(n)
                else:
                    LIS[i] = n 
            return len(LIS)
        return LIS(nums) >= 3
```

- math

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [inf] * 3
        for i, v in enumerate(nums):
            if v < f[1]:
                f[1] = v 
            elif f[1] < v < f[2]:
                f[2] = v 
            elif v > f[2]:
                return True
        return False
```

### 646. Maximum Length of Pair Chain

```python
class Solution:
    def findLongestChain(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        n = len(intervals)
        f = [1] * n
        for i in range(1, n):
            for j in range(i):
                if intervals[i][0] > intervals[j][1]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)
```

```python
class Solution:
    def findLongestChain(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        n = len(intervals)
        def LIS(arr): # strictly increasing
            LIS = []
            for x, y in arr:
                i = bisect_left(LIS, x)
                if i == len(LIS):
                    LIS.append(y)
                else:
                    LIS[i] = min(LIS[i], y) # insert need to check min y
            return len(LIS) 
        return LIS(intervals)
```

### 1626. Best Team With No Conflicts

```python
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        a = sorted(zip(scores, ages))
        f = [0] * len(a)
        for i, (score, age) in enumerate(a):
            for j in range(i):
                if a[j][1] <= age:
                    f[i] = max(f[i], f[j])
            f[i] += score
        return max(f)
```

### 354. Russian Doll Envelopes

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key = lambda x: [x[0], -x[1]])
        def LIS(arr): # strictly increasing
            LIS = []
            for n in arr:
                i = bisect_left(LIS, n)
                if i == len(LIS):
                    LIS.append(n)
                else:
                    LIS[i] = n
            return len(LIS)  
        res = [b for a, b in envelopes] 
        return LIS(res)
```

### 1691. Maximum Height by Stacking Cuboids 

```python
class Solution:
    def maxHeight(self, cuboids: List[List[int]]) -> int:
        for c in cuboids:
            c.sort()
        cuboids.sort()
        f = [0] * len(cuboids)
        for i, (w1, l1, h1) in enumerate(cuboids):
            for j, (w2, l2, h2) in enumerate(cuboids[:i]):
                if l2 <= l1 and h2 <= h1:
                    f[i] = max(f[i], f[j])
            f[i] += h1 
        return max(f)
```

### 1027. Longest Arithmetic Subsequence

```python
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        res = 2
        for diff in range(-500, 501):
            dp = defaultdict(int)
            for n in nums:
                dp[n] = dp[n - diff] + 1
            res = max(res, max(dp.values()))
        return res
```