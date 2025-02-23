# DP

## 1 Basics (23)

### 1.1 Climbing Stairs (8)

* [70. Climbing Stairs](#70-climbing-stairs) 1300
* [509. Fibonacci Number](#509-fibonacci-number) 1300
* [1137. N-th Tribonacci Number](#1137-n-th-tribonacci-number) 1400
* [746. Min Cost Climbing Stairs](#746-min-cost-climbing-stairs) 1500
* [377. Combination Sum IV](#377-combination-sum-iv) 1600
* [2466. Count Ways To Build Good Strings](#2466-count-ways-to-build-good-strings) 1694
* [2533. Number of Good Binary Strings](#2533-number-of-good-binary-strings) 1694
* [2266. Count Number of Texts](#2266-count-number-of-texts) 1857

### 1.2 House Robber (8)

- f[i] = max(f[i - 1], f[i] + f[i - 2])

* [198. House Robber](#198-house-robber) 1500
* [740. Delete and Earn](#740-delete-and-earn) 1600
* [2320. Count Number of Ways to Place Houses](#2320-count-number-of-ways-to-place-houses) 1608
* [213. House Robber II](#213-house-robber-ii) 1650
* [3186. Maximum Total Damage With Spell Casting](#3186-maximum-total-damage-with-spell-casting) 1840
* [256. Paint House](#256-paint-house)
* [265. Paint House II](#265-paint-house-ii)
* [276. Paint Fence](#276-paint-fence)

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
        for n in range(3, n + 1):
            second, first = second + first, second 
        return second if n >= 2 else 1
```

### 509. Fibonacci Number

```python
class Solution:
    def fib(self, n: int) -> int:
        first, second = 0, 1
        for i in range(2, n + 1):
            second, first = first + second, second
        return second if n >= 1 else 0
```

### 1137. N-th Tribonacci Number

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        first, second, third = 0, 1, 1
        for i in range(3, n + 1):
            third, second, first = third + second + first, third, second 
        return third if n >= 1 else 0
```

### 746. Min Cost Climbing Stairs

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        for i in range(2, n):
            cost[i] += min(cost[i - 2], cost[i - 1])
        return min(cost[-2:])
```

### 377. Combination Sum IV 

- (unbounded knapsack)

- climbing stairs idea

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @cache
        def f(x):
            if x == target:
                return 1
            if x > target:
                return 0
            res = 0
            for n in nums:
                res += f(x + n)
            return res 
        return f(0)
```

### 2466. Count Ways To Build Good Strings

- same as 2533 (unbounded knapsack), use t from low to high is better

```python
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def f(t):
            if t == 0:
                return 1
            if t < 0:
                return 0
            res = 0
            for n in [zero, one]:
                res = (res + f(t - n)) % mod
            return res 
        return sum(f(i) for i in range(low, high + 1)) % mod
```

### 2533. Number of Good Binary Strings

- same as 2466 (unbounded knapsack)

```python
class Solution:
    def goodBinaryStrings(self, minLength: int, maxLength: int, oneGroup: int, zeroGroup: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def f(t):
            if t == 0:
                return 1
            if t < 0:
                return 0
            res = 0
            for x in [oneGroup, zeroGroup]:
                res = (res + f(t - x)) % mod
            return res
        return sum(f(i) for i in range(minLength, maxLength + 1)) % mod 
```

### 2266. Count Number of Texts

```python
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        mod = 10 ** 9 + 7
        def f(c, count):
            first, second, third, fourth = 1, 1, 2, 4
            for _ in range(4, count + 1):
                if c in '234568':
                    fourth, third, second, first = fourth + third + second, fourth, third, second 
                else:
                    fourth, third, second, first = fourth + third + second + first, fourth, third, second 
            return fourth % mod 
        res = 1
        for c, s in groupby(pressedKeys):
            count = len(list(s))
            if count <= 2:
                res = (res * count) % mod 
            else:
                res = (res * f(c, count)) % mod 
        return res 
```

### 1.2 House Robber (8)

- f[i] = max(f[i - 1], f[i] + f[i - 2])

* [198. House Robber](#198-house-robber) 1500
* [740. Delete and Earn](#740-delete-and-earn) 1600
* [2320. Count Number of Ways to Place Houses](#2320-count-number-of-ways-to-place-houses) 1608
* [213. House Robber II](#213-house-robber-ii) 1650
* [3186. Maximum Total Damage With Spell Casting](#3186-maximum-total-damage-with-spell-casting) 1840
* [256. Paint House](#256-paint-house)
* [265. Paint House II](#265-paint-house-ii)
* [276. Paint Fence](#276-paint-fence)

### 198. House Robber

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        nums = [0] + nums
        n = len(nums)
        for i in range(2, n):
            nums[i] = max(nums[i - 1], nums[i] + nums[i - 2])
        return nums[-1]
```

### 740. Delete and Earn

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        f = [0] * (max(nums) + 1)
        for n in nums:
            f[n] += n 
        f = [0] + f 
        for i in range(2, len(f)):
            f[i] = max(f[i - 1], f[i - 2] + f[i])
        return f[-1]
```

### 2320. Count Number of Ways to Place Houses

```python
class Solution:
    def countHousePlacements(self, n: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def f(i, rob):
            if i == n:
                return 1
            res = 0
            if rob:
                res += f(i + 1, not rob)
            else:
                res += f(i + 1, rob) + f(i + 1, not rob)
            return res 
        res = f(-1, True) % mod 
        return (res * res) % mod
```

### 256. Paint House

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        n = len(costs)
        for i in range(1, n):
            for j in range(3):
                costs[i][j] += min(costs[i - 1][:j] + costs[i - 1][j + 1:])
        return min(costs[-1])
```

### 265. Paint House II

```python
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        n = len(costs)
        for i in range(1, n):
            for j in range(len(costs[0])):
                costs[i][j] += min(costs[i - 1][:j] + costs[i - 1][j + 1:])
        return min(costs[-1])
```

### 213. House Robber II

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def rob1(nums):
            nums = [0] + nums 
            n = len(nums)
            for i in range(2, n):
                nums[i] = max(nums[i - 1], nums[i] + nums[i - 2])
            return nums[-1]
        return max(rob1(nums[1:]), rob1(nums[:-1])) if len(nums) > 1 else nums[0]
```

### 3186. Maximum Total Damage With Spell Casting

```python
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        cnt = Counter(power)
        a = sorted(cnt.keys())
        @cache
        def dfs(i):
            if i < 0:
                return 0 
            x = a[i]
            j = i - 1 # a number before current
            while j >= 0 and x - a[j] <= 2: # meet the condition
                j -= 1 # final j didn't meet the condition and use dfs(j)
            return max(dfs(i - 1), dfs(j) + x * cnt[x])
        return dfs(len(a) - 1)
```

### 276. Paint Fence

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        @cache
        def dfs(i):
            if i == 1:
                return k 
            if i == 2:
                return k * k
            return (dfs(i - 1) + dfs(i - 2)) * (k - 1)
        return dfs(n)
```

### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(1, n):
            nums[i] = max(nums[i], nums[i - 1] + nums[i])
        return max(nums)
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
        def max_subarray_sum(nums):
            n = len(nums)
            for i in range(1, n):
                nums[i] = max(nums[i], nums[i - 1] + nums[i])
            return max(0, *nums)
        
        if k == 1:
            return max_subarray_sum(arr) % mod 
        total = sum(arr)
        arr2 = arr * 2
        res = max_subarray_sum(arr2)
        if total > 0:
            return ((k - 2) * total + res) % mod
        return res % mod
```

### 918. Maximum Sum Circular Subarray

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        def max_subarray_sum(nums):
            n = len(nums)
            for i in range(1, n):
                nums[i] = max(nums[i], nums[i - 1] + nums[i])
            return max(nums)
        def min_subarray_sum(nums):
            n = len(nums)
            for i in range(1, n):
                nums[i] = min(nums[i], nums[i - 1] + nums[i])
            return min(nums)
        total = sum(nums)
        if all([n < 0 for n in nums]):
            return max(nums)
        return max(max_subarray_sum(nums[::]), total - min_subarray_sum(nums[::]))
```

### 2321. Maximum Score Of Spliced Array

```python
class Solution:
    def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
        arr1 = [n1 - n2 for n1, n2 in zip(nums1, nums2)]
        arr2 = [n2 - n1 for n1, n2 in zip(nums1, nums2)]
        n = len(arr1)
        for i in range(1, n):
            arr1[i] = max(arr1[i], arr1[i - 1] + arr1[i])
            arr2[i] = max(arr2[i], arr2[i - 1] + arr2[i])
        return max(max(arr1) + sum(nums2), max(arr2) + sum(nums1))
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
