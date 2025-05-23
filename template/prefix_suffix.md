## template

```python
class Solution:
    def minimumSum(self, nums: List[int]) -> int: # 2909
        n, res = len(nums), inf
        prefix, suffix = nums[::], nums[::]
        for i in range(1, n):
            prefix[i] = min(prefix[i], prefix[i - 1])
        for i in range(n - 2, -1, -1):
            suffix[i] = min(suffix[i], suffix[i + 1])
        for i in range(1, n - 1):
            if prefix[i - 1] < nums[i] and nums[i] > suffix[i + 1]:
                res = min(res, nums[i] + prefix[i - 1] + suffix[i + 1])
        return res if res != inf else -1
```

### prefix suffix(10)

* [238. Product of Array Except Self](#238-product-of-array-except-self)
* [2906. Construct Product Matrix](#209-Minimum-Size-Subarray-Sum)
* [2483. Minimum Penalty for a Shop](#2483-minimum-penalty-for-a-shop)
* [334. Increasing Triplet Subsequence](#334-increasing-triplet-subsequence)
* [2256. Minimum Average Difference](#2256-minimum-average-difference)
* [42. Trapping Rain Water](#42-trapping-rain-water)
* [2909. Minimum Sum of Mountain Triplets II](#2906-construct-product-matrix)
* [2055. Plates Between Candles](#2055-plates-between-candles)
* [2012. Sum of Beauty in the Array](#2012-sum-of-beauty-in-the-array)
* [915. Partition Array into Disjoint Intervals](#915-partition-array-into-disjoint-intervals)

### 334. Increasing Triplet Subsequence

- prefix suffix

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        prefix, suffix = nums[::], nums[::]
        for i in range(1, n):
            prefix[i] = min(prefix[i], prefix[i - 1])
        for i in range(n - 2, -1, -1):
            suffix[i] = max(suffix[i], suffix[i + 1])
        for i in range(1, n - 1):
            if prefix[i] < nums[i] < suffix[i]:
                return True
        return False
```

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

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        def LIS(nums):
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

### 42. Trapping Rain Water

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        def prefix(height):
            res = height[::]
            for i in range(1, len(res)):
                res[i] = max(res[i - 1], height[i])
            return res
        prefix, suffix = prefix(height), prefix(height[::-1])[::-1]
        return sum(min(l, r) - v for l, r, v in zip(prefix, suffix, height))
```

### 238. Product of Array Except Self

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n

        suf = 1
        for i in range(n - 1, -1, -1):
            res[i] = suf 
            suf = suf * nums[i]
        
        pre = 1
        for i in range(n):
            res[i] = pre * res[i]
            pre = pre * nums[i]

        return res
```

### 2906. Construct Product Matrix

```python
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        res =[[0] * C for r in range(R)]
        mod = 12345

        suf = 1
        for r in range(R - 1, -1, -1):
            for c in range(C - 1, -1, -1):
                res[r][c] = suf
                suf = suf * grid[r][c] % mod

        pre = 1
        for r in range(R):
            for c in range(C):
                res[r][c] = pre * res[r][c] % mod
                pre = pre * grid[r][c] % mod

        return res
```

### 2256. Minimum Average Difference

```python
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        n = len(nums)
        res = [0] * n

        suf = 0
        for i in range(n - 1, -1, -1):
            res[i] = suf
            suf = suf + nums[i]
            
        pre = 0
        ans, idx = inf, 0
        for i in range(n):
            pre = pre + nums[i]
            if n - i - 1 == 0:
                res[i] = int(abs(pre / (i + 1)))
            else:
                res[i] = abs(pre // (i + 1) - (res[i] // (n - i - 1)))
            if res[i] < ans:
                ans = res[i]
                idx = i
        return idx
```

### 2483. Minimum Penalty for a Shop

```python
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        n = len(customers)
        res = [0] * n

        suf = 0
        for i in range(n - 1, -1, -1):
            if customers[i] == 'Y':
                suf += 1
            res[i] = suf

        pre = 0
        for i in range(n):
            if i > 0 and customers[i - 1] == 'N':
                pre += 1
            res[i] = pre + res[i]
        res = res + [pre] if customers[n - 1] != 'N' else res + [pre + 1]
        minNum = min(res)
        for i, n in enumerate(res):
            if n == minNum:
                return i
```

### 2909. Minimum Sum of Mountain Triplets II

```python
class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n, res = len(nums), inf
        prefix, suffix = nums[::], nums[::]
        for i in range(1, n):
            prefix[i] = min(prefix[i], prefix[i - 1])
        for i in range(n - 2, -1, -1):
            suffix[i] = min(suffix[i], suffix[i + 1])
        for i in range(1, n - 1):
            if prefix[i - 1] < nums[i] and nums[i] > suffix[i + 1]:
                res = min(res, nums[i] + prefix[i - 1] + suffix[i + 1])
        return res if res != inf else -1
```

### 2420. Find All Good Indices

```python
class Solution:
    def goodIndices(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        prefix, suffix = [False] * n, [False] * n
        pre1, pre2, count1, count2 = inf, -inf, 0, 0
        for i in range(n):
            if nums[i] <= pre1:
                count1 += 1
            else:
                count1 = 1
            pre1 = nums[i]
            if count1 >= k and i + 1 < n:
                prefix[i + 1] = True
            if nums[i] >= pre2:
                count2 += 1
            else:
                count2 = 1
            pre2 = nums[i]
            if count2 >= k and i - k >= 0:
                suffix[i - k] = True
        
        return [i for i in range(n) if prefix[i] and suffix[i]]
```

### 2167. Minimum Time to Remove All Cars Containing Illegal Goods

```python
class Solution:
    def minimumTime(self, s: str) -> int:
        n = len(s)
        prefix, suffix = [0] * (n + 1), [0] * (n + 1)
        for i in range(1, n + 1):
            if s[i - 1] == '1':
                prefix[i] = min(prefix[i - 1] + 2, i)
            else:
                prefix[i] = prefix[i - 1]
        for i in range(n - 1, -1, -1):
            if s[i] == '1':
                suffix[i] = min(suffix[i + 1] + 2, n - i)
            else:
                suffix[i] = suffix[i + 1] 
        return min(a + b for a, b in zip(prefix, suffix))
```

### 2484. Count Palindromic Subsequences

```python
MOD = 10 ** 9 + 7

class Solution:
    def countPalindromes(self, s: str) -> int:
        n = len(s)
        pre = [[[0] * (n + 1) for _ in range(10)] for _ in range(10)]
        suf = [[[0] * (n + 1) for _ in range(10)] for _ in range(10)]
        for i in range(10):
            for j in range(10):
                cnt = 0
                for k in range(1, n + 1):
                    pre[i][j][k] = pre[i][j][k - 1]
                    if s[k - 1] == str(j):
                        pre[i][j][k] += cnt
                    if s[k - 1] == str(i):
                        cnt += 1
                cnt = 0
                for k in range(n - 1, -1, -1):
                    suf[i][j][k] = suf[i][j][k + 1]
                    if s[k] == str(i):
                        suf[i][j][k] += cnt
                    if s[k] == str(j):
                        cnt += 1
                        
        ans = 0
        for k in range(n - 2):
            for i in range(10):
                for j in range(10):
                    ans += pre[i][j][k] * suf[j][i][k + 1]
        return ans % MOD
```

* [2104. Sum of Subarray Ranges](#2104-sum-of-subarray-ranges)

### 2104. Sum of Subarray Ranges

```python
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        n, res = len(nums), 0
        for i in range(n):
            mn, mx = nums[i], nums[i]
            for j in range(i, n):
                mn = min(mn, nums[j])
                mx = max(mx, nums[j])
                res += mx - mn
        return res
```

### 2012. Sum of Beauty in the Array

```python
class Solution:
    def sumOfBeauties(self, nums: List[int]) -> int:
        preMax, sufMin = nums[::], nums[::]
        n = len(nums)
        for i in range(1, n):
            preMax[i] = max(preMax[i], preMax[i - 1])
        for i in range(n - 2, -1, -1):
            sufMin[i] = min(sufMin[i], sufMin[i + 1])
        res = 0
        for i in range(1, n - 1):
            if preMax[i - 1] < nums[i] and nums[i] < sufMin[i + 1]:
                res += 2
            elif nums[i - 1] < nums[i] < nums[i + 1]:
                res += 1
            else:
                res += 0
        return res
```

### 2055. Plates Between Candles

```python
class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        right_closest, left_closest = inf, inf 
        n = len(s)
        left = [0] * n
        for i, c in enumerate(s):
            if c == '|':
                left_closest = i 
            left[i] = left_closest
        right = [0] * n
        for i in range(n - 1, -1, -1):
            if s[i] == '|':
                right_closest = i 
            right[i] = right_closest
        
        pre = [0] * n
        for i, c in enumerate(s):
            if c == '*':
                pre[i] = 1
        pre = list(accumulate(pre, initial = 0))
        res = [0] * len(queries)
        for i, (a, b) in enumerate(queries):
            if left[b] != inf and right[a] != inf and pre[left[b] + 1] - pre[right[a]] > 0:
                res[i] = pre[left[b] + 1] - pre[right[a]]
        return res
```

### 915. Partition Array into Disjoint Intervals

```python
class Solution:
    def partitionDisjoint(self, nums: List[int]) -> int:
        preMax = nums[::]
        sufMax = nums[::]
        for i in range(1, len(nums)):
            preMax[i] = max(preMax[i - 1], preMax[i])
        for i in range(len(nums) - 2, -1, -1):
            sufMax[i] = min(sufMax[i + 1], sufMax[i])
        for i in range(len(nums) - 1):
            if preMax[i] <= sufMax[i + 1]:
                return i + 1
```

### ### 2680. Maximum OR

```python
class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pre = nums[::]
        suf = nums[::]
        for i in range(1, n):
            pre[i] |= pre[i - 1]
        for i in range(n - 2, -1, -1):
            suf[i] |= suf[i + 1]
        pre = [0] + pre + [0]
        suf = [0] + suf + [0]
        res = 0
        for i in range(0, n):
            res = max(res, pre[i] | suf[i + 2] | (nums[i] << k))
        return res 
```

### 2100. Find Good Days to Rob the Bank

```python 
class Solution:
    def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
        n = len(security)
        pre = [1] * n
        suf = [1] * n
        for i in range(1, n):
            if security[i] <= security[i - 1]:
                pre[i] += pre[i - 1]
        for i in range(n - 2, -1, -1):
            if security[i] <= security[i + 1]:
                suf[i] += suf[i + 1]
        res = []
        for i in range(n):
            if pre[i] >= time + 1 and suf[i] >= time + 1:
                res.append(i)
        return res
```