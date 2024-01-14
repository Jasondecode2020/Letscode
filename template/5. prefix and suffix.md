## template

- prefix suffix

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

* [238. Product of Array Except Self](#3-Longest-Substring-Without-Repeating-Characters)
* [2906. Construct Product Matrix](#209-Minimum-Size-Subarray-Sum)
* [2483. Minimum Penalty for a Shop](#15-3Sum)
* [334. Increasing Triplet Subsequence](#3-Longest-Substring-Without-Repeating-Characters)
* [2256. Minimum Average Difference](#209-Minimum-Size-Subarray-Sum)
* [42. Trapping Rain Water](#15-3Sum)
* [2909. Minimum Sum of Mountain Triplets II](#15-3Sum)

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