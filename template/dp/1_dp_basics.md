# DP

## 1 Basics (19)

### 1.1 Climbing Stairs (8)

* [70. Climbing Stairs](#70-climbing-stairs) 1300
* [509. Fibonacci Number](#509-fibonacci-number) 1300
* [1137. N-th Tribonacci Number](#1137-n-th-tribonacci-number) 1400
* [746. Min Cost Climbing Stairs](#746-min-cost-climbing-stairs) 1500
* [377. Combination Sum IV](#377-combination-sum-iv) 1600
* [2466. Count Ways To Build Good Strings](#2466-count-ways-to-build-good-strings) 1694
* [2266. Count Number of Texts](#2266-count-number-of-texts) 1857
* [2533. Number of Good Binary Strings](#2533-number-of-good-binary-strings) 1694

### 1.2 House Robber (4)

* [198. House Robber](#198-house-robber) 1500
* [740. Delete and Earn](#740-delete-and-earn) 1600
* [2320. Count Number of Ways to Place Houses](#2320-count-number-of-ways-to-place-houses) 1608
* [213. House Robber II](#213-house-robber-ii) 1650

### 1.3 maximum Subarray sum (7)

- f[i] = f[i−1] + a[i]

* [53. Maximum Subarray](#53-maximum-subarray)
* [2606. Find the Substring With Maximum Cost](#2606-find-the-substring-with-maximum-cost) 1422
* [1749. Maximum Absolute Sum of Any Subarray](#1749-maximum-absolute-sum-of-any-subarray) 1542
* [1191. K-Concatenation Maximum Sum](#1191-k-concatenation-maximum-sum) 1748
* [918. Maximum Sum Circular Subarray](#918-maximum-sum-circular-subarray) 1777
* [2321. Maximum Score Of Spliced Array](#2321-maximum-score-of-spliced-array) 1791
* [152. Maximum Product Subarray](#152-maximum-product-subarray)

### 70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        first, second = 1, 2
        for i in range(3, n + 1):
            second, first = second + first, second 
        return second if n >= 2 else first 
```

### 509. Fibonacci Number

```python
class Solution:
    def fib(self, n: int) -> int:
        first, second = 0, 1
        for i in range(2, n + 1):
            second, first = second + first, second 
        return second if n >= 1 else first 
```

### 1137. N-th Tribonacci Number

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        first, second, third = 0, 1, 1
        for i in range(3, n + 1):
            third, second, first = third + second + first, third, second 
        return third if n >= 2 else n
```

### 746. Min Cost Climbing Stairs

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        for i in range(2, n):
            cost[i] += min(cost[i - 2: i])
        return min(cost[-2:])
```

### 377. Combination Sum IV 

- (unbounded knapsack)

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @cache
        def dfs(t):
            if t > target:
                return 0
            if t == target:
                return 1
            res = 0
            for n in nums:
                res += dfs(n + t)
            return res 
        return dfs(0)
```

### 2466. Count Ways To Build Good Strings

- same as 2466 (unbounded knapsack)

```python
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(t):
            if t == 0:
                return 1
            if t < 0:
                return 0 
            res = 0
            for n in [zero, one]:
                res = (res + dfs(t - n)) % mod
            return res % mod
        return sum(dfs(t) for t in range(high, low - 1, -1)) % mod
```

### 2266. Count Number of Texts

```python
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        mod = 10 ** 9 + 7 
        @cache
        def dp(c, count):
            first, second, third, fourth = 1, 1, 2, 4 
            for _ in range(4, count + 1):
                if c in '234568':
                    fourth, third, second = fourth + third + second, fourth, third 
                else:
                    fourth, third, second, first = fourth + third + second + first, fourth, third, second
            if count <= 2:
                return count 
            return fourth 

        res = 1
        for c, s in groupby(pressedKeys):
            count = len(list(s))
            res = (res * dp(c, count)) % mod 
        return res 
```

```python
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        mod = 10 ** 9 + 7
        arr = []
        i = 0
        n = len(pressedKeys)
        while i < n:
            j = i
            while j < n and pressedKeys[j] == pressedKeys[i]:
                j += 1
            arr.append((pressedKeys[i], j - i))
            i = j
        
        def dp(c, count):
            first, second, third, fourth = 1, 1, 2, 4
            for i in range(4, count + 1):
                if c in '234568':
                    fourth, third, second = fourth + third + second, fourth, third
                else:
                    fourth, third, second, first = fourth + third + second + first, fourth, third, second
            if count <= 1:
                return second
            elif count == 2:
                return third
            elif count >= 3:
                return fourth % mod
        res = 1
        for c, count in arr:
            res = (res * dp(c, count)) % mod
        return res
```

### 2533. Number of Good Binary Strings

- same as 2466 (unbounded knapsack)

```python
class Solution:
    def goodBinaryStrings(self, minLength: int, maxLength: int, oneGroup: int, zeroGroup: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(t):
            if t == 0:
                return 1
            if t < 0:
                return 0
            res = 0
            for n in [oneGroup, zeroGroup]:
                res = (res + dfs(t - n)) % mod 
            return res 
        return sum(dfs(t) for t in range(minLength, maxLength + 1)) % mod
```

### 1.2 House Robber (4)

* [198. House Robber](#198-house-robber)
* [740. Delete and Earn](#740-delete-and-earn)
* [2320. Count Number of Ways to Place Houses](#2320-count-number-of-ways-to-place-houses) 1608
* [213. House Robber II](#213-house-robber-ii)

### 198. House Robber

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] + nums 
        for i in range(2, n + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[-1]
```

### 740. Delete and Earn

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        dp = [0] * (max(nums) + 1)
        for n in nums:
            dp[n] += n 

        for i in range(2, len(dp)):
            dp[i] = max(dp[i - 1], dp[i] + dp[i - 2])
        return dp[-1]
```

### 2320. Count Number of Ways to Place Houses

```python
class Solution:
    def countHousePlacements(self, n: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, rob):
            if i == n - 1:
                return 1
            res = 0
            if rob:
                res += dfs(i + 1, not rob)
            else:
                res += dfs(i + 1, rob) + dfs(i + 1, not rob)
            return res % mod
        res = (dfs(0, True) + dfs(0, False)) % mod
        return (res * res) % mod
```

### 213. House Robber II

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def rob_a_line(nums):
            dp = [0] + nums 
            for i in range(2, len(dp)):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
            return dp[-1]
        if len(nums) == 1:
            return nums[0]
        return max(rob_a_line(nums[1: ]), rob_a_line(nums[: -1]))
```

### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [nums[0]] * n 
        for i in range(1, n):
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
        return max(dp)
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        def dfs(l, r):
            if l == r:
                return nums[l]
            m = l + (r - l) // 2
            res, left = 0, -inf
            for i in range(m, l - 1, -1):
                res += nums[i]
                left = max(left, res)

            res, right = 0, -inf
            for i in range(m + 1, r + 1):
                res += nums[i]
                right = max(right, res)
            return max(dfs(l, m), dfs(m + 1, r), left + right)
        return dfs(0, len(nums) - 1)
```

### 2606. Find the Substring With Maximum Cost

```python
class Solution:
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        n = len(s)
        d = Counter()
        for c, v in zip(chars, vals):
            d[c] = v 
        nums = [0] * n 
        for i, c in enumerate(s):
            if c not in d:
                nums[i] = ord(c) - ord('a') + 1
            else:
                nums[i] = d[c]
        dp = [nums[0]] * n 
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(0, max(dp))
```

### 1749. Maximum Absolute Sum of Any Subarray

```python
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        mx, mn, res = 0, 0, 0
        for n in nums:
            mx = max(mx + n, n)
            mn = min(mn + n, n)
            res = max(res, mx, abs(mn))
        return res
```

### 1191. K-Concatenation Maximum Sum

```python
class Solution:
    def kConcatenationMaxSum(self, arr: List[int], k: int) -> int:
        mod = 10 ** 9 + 7
        def max_sum(arr):
            n = len(arr)
            dp = [arr[0]] * n 
            for i in range(1, n):
                dp[i] = max(arr[i], dp[i - 1] + arr[i])
            return max(dp + [0])
        if k == 1:
            return max_sum(arr) % mod 
        total = sum(arr)
        concat_sum = max_sum(arr * 2)
        if total > 0:
            return max(concat_sum, total * (k - 2) + concat_sum) % mod
        else:
            return concat_sum % mod
```

### 918. Maximum Sum Circular Subarray

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        maxNum, minNum = [nums[0]] * n, [nums[0]] * n
        for i in range(1, n):
            maxNum[i] = max(nums[i], nums[i] + maxNum[i - 1])
        for i in range(1, n):
            minNum[i] = min(nums[i], nums[i] + minNum[i - 1])
        if sum(nums) - min(minNum) == 0:
            return max(maxNum)
        return max(max(maxNum), sum(nums) - min(minNum))
```

### 2321. Maximum Score Of Spliced Array

```python
class Solution:
    def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
        arr1 = [n1 - n2 for n1, n2 in zip(nums1, nums2)]
        arr2 = [n2 - n1 for n1, n2 in zip(nums1, nums2)]
        n = len(arr1)
        pre1, pre2 = [arr1[0]] * n, [arr2[0]] * n
        for i in range(1, n):
            pre1[i] = max(arr1[i], pre1[i - 1] + arr1[i])
            pre2[i] = max(arr2[i], pre2[i - 1] + arr2[i])
        return max(max(pre1) + sum(nums2), max(pre2) + sum(nums1))
```

### 152. Maximum Product Subarray

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = mx = mn = nums[0]
        n = len(nums)
        for i in range(1, n):
            temp = mx 
            mx = max(mx * nums[i], mn * nums[i], nums[i])
            mn = min(temp * nums[i], mn * nums[i], nums[i])
            res = max(res, mx)
        return res
```
