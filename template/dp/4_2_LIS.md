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
def lis(arr): # more than or equal
    f = []
    for n in arr:
        i = bisect_right(f, n)
        if i == len(f):
            f.append(n)
        else:
            f[i] = n
    return len(f)   
```

```python
def lis(arr): # strictly increasing
    f = []
    for n in arr:
        i = bisect_left(f, n)
        if i == len(f):
            f.append(n)
        else:
            f[i] = n
    return len(f)
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

### question list (17)

* [300. Longest Increasing Subsequence](#300-longest-increasing-subsequence)
* [334. Increasing Triplet Subsequence](#334-increasing-triplet-subsequence)
* [646. Maximum Length of Pair Chain](#646-maximum-length-of-pair-chain)
* [673. Number of Longest Increasing Subsequence](#673-number-of-longest-increasing-subsequence)
* [2826. Sorting Three Groups](#2826-sorting-three-groups)

* [1671. Minimum Number of Removals to Make Mountain Array](#1671-minimum-number-of-removals-to-make-mountain-array)
* [1964. Find the Longest Valid Obstacle Course at Each Position](#1964-find-the-longest-valid-obstacle-course-at-each-position)
* [1626. Best Team With No Conflicts](#1626-best-team-with-no-conflicts)
* [354. Russian Doll Envelopes](#354-russian-doll-envelopes)
* [1691. Maximum Height by Stacking Cuboids](#1691-maximum-height-by-stacking-cuboids)

* [1027. Longest Arithmetic Subsequence](#1027-longest-arithmetic-subsequence)
* [2770. Maximum Number of Jumps to Reach the Last Index](#2770-maximum-number-of-jumps-to-reach-the-last-index)
* [2111. Minimum Operations to Make the Array K-Increasing](#2111-minimum-operations-to-make-the-array-k-increasing)
* [960. Delete Columns to Make Sorted III](#960-delete-columns-to-make-sorted-iii)
* [2407. Longest Increasing Subsequence II](#2407-longest-increasing-subsequence-ii)

* [1187. Make Array Strictly Increasing](#1187-make-array-strictly-increasing)
* [1713. Minimum Operations to Make a Subsequence](#1713-minimum-operations-to-make-a-subsequence)

### 300. Longest Increasing Subsequence

- binary search: O(nlog(n))

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def lis(nums):
            f = []
            for n in nums:
                i = bisect_left(f, n)
                if i == len(f):
                    f.append(n)
                else:
                    f[i] = n 
            return len(f)
        return lis(nums)
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
        def lis(nums):
            f = []
            for n in nums:
                i = bisect_left(f, n)
                if i == len(f):
                    f.append(n)
                else:
                    f[i] = n 
            return len(f)
        return lis(nums) >= 3
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
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        n = len(pairs)
        f = [1] * n 
        pairs.sort()
        for i in range(n):
            for j in range(i):
                if pairs[j][1] < pairs[i][0]:
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

### 673. Number of Longest Increasing Subsequence

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        count = [1] + [0] * (n - 1)
        for r in range(1, n):
            for l in range(r):
                if nums[r] > nums[l]:
                    dp[r] = max(dp[r], dp[l] + 1)
            for l in range(r):
                if nums[r] > nums[l] and dp[l] == dp[r] - 1:
                    count[r] += count[l]
            if not count[r]:
                count[r] = 1

        mx = max(dp)
        return sum(count[i] for i, n in enumerate(dp) if n == mx)
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

### 2770. Maximum Number of Jumps to Reach the Last Index

```python
class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-inf] * n 
        dp[0] = 0
        for j in range(1, n):
            for i in range(j - 1, -1, -1):
                if abs(nums[j] - nums[i]) <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        return dp[-1] if dp[-1] >= 0 else -1
```

### 2111. Minimum Operations to Make the Array K-Increasing

```python
class Solution:
    def kIncreasing(self, arr: List[int], k: int) -> int:
        def lis(nums):
            f = []
            for n in nums:
                i = bisect_right(f, n)
                if i == len(f):
                    f.append(n)
                else:
                    f[i] = n 
            return len(f)

        res = 0
        for i in range(k):
            nums = arr[i::k]
            res += len(nums) - lis(nums)
        return res
```

### 2826. Sorting Three Groups

```python
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        def lis(nums):
            f = []
            for n in nums:
                i = bisect_right(f, n)
                if i == len(f):
                    f.append(n)
                else:
                    f[i] = n 
            return len(f)
        return len(nums) - lis(nums)
```

### 1671. Minimum Number of Removals to Make Mountain Array

```python
'''
LIS with prefix suffix
'''
class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        def lis(nums):
            n = len(nums)
            f = [1] * n 
            for i in range(1, n):
                for j in range(i):
                    if nums[j] < nums[i]:
                        f[i] = max(f[i], f[j] + 1)
            return f 
        pre = lis(nums)
        suf = lis(nums[::-1])[::-1]
        res = 0
        for i, (p, s) in enumerate(zip(pre, suf)):
            if p > 1 and s > 1:
                res = max(res, p + s - 1)
        return len(nums) - res 
```

### 1964. Find the Longest Valid Obstacle Course at Each Position

```python
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        def lis(nums):
            res = [0] * len(nums)
            f = []
            for i, n in enumerate(nums):
                j = bisect_right(f, n)
                if j == len(f):
                    f.append(n)
                else:
                    f[j] = n 
                res[i] = j if j == len(f) else j + 1
            return res
        return lis(obstacles)
```

### 2407. Longest Increasing Subsequence II

```python
class Solution:
    def lengthOfLIS(self, nums: List[int], k: int) -> int:
        u = max(nums)
        mx = [0] * (4 * u)

        def modify(o: int, l: int, r: int, i: int, val: int) -> None:
            if l == r:
                mx[o] = val
                return
            m = (l + r) // 2
            if i <= m: modify(o * 2, l, m, i, val)
            else: modify(o * 2 + 1, m + 1, r, i, val)
            mx[o] = max(mx[o * 2], mx[o * 2 + 1])

        # 返回区间 [L,R] 内的最大值
        def query(o: int, l: int, r: int, L: int, R: int) -> int:  # L 和 R 在整个递归过程中均不变，将其大写，视作常量
            if L <= l and r <= R: return mx[o]
            res = 0
            m = (l + r) // 2
            if L <= m: res = query(o * 2, l, m, L, R)
            if R > m: res = max(res, query(o * 2 + 1, m + 1, r, L, R))
            return res

        for x in nums:
            if x == 1:
                modify(1, 1, u, 1, 1)
            else:
                res = 1 + query(1, 1, u, max(x - k, 1), x - 1)
                modify(1, 1, u, x, res)
        return mx[1]
```

### 1187. Make Array Strictly Increasing

```python
class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        arr2.sort()
        @cache
        def dfs(i, next):
            if i < 0:
                return 0
            res = dfs(i - 1, arr1[i]) if arr1[i] < next else inf
            j = bisect_left(arr2, next) - 1
            if j >= 0:
                res = min(res, dfs(i - 1, arr2[j]) + 1)
            return res
        res = dfs(len(arr1) - 1, inf)
        return res if res != inf else -1
```

### 1713. Minimum Operations to Make a Subsequence

```python
class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        # target = [5,1,3], arr = [5,9,4,2,3,4] [0, 2]
        # d = {5: 0, 1: 1, 3: 2}
        d = {n: i for i, n in enumerate(target)}
        def LIS(arr): # strictly increasing
            LIS = []
            for n in arr:
                if n in d:
                    i = bisect_left(LIS, d[n])
                    if i == len(LIS):
                        LIS.append(d[n])
                    else:
                        LIS[i] = d[n]
            return len(LIS)   
        return len(target) - LIS(arr)
```

### 960. Delete Columns to Make Sorted III

```python
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        n = len(strs[0])
        f = [1] * n
        for i in range(1, n):
            for j in range(i):
                if all(row[j] <= row[i] for row in strs):
                    f[i] = max(f[i], f[j] + 1)
        return n - max(f)
```