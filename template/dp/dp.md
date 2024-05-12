# DP

## 1 Basics

### 1.1 Climbing Stairs (8)

* [70. Climbing Stairs](#70-climbing-stairs)
* [509. Fibonacci Number](#509-fibonacci-number)
* [1137. N-th Tribonacci Number](#1137-n-th-tribonacci-number)
* [746. Min Cost Climbing Stairs](#746-min-cost-climbing-stairs)
* [377. Combination Sum IV](#377-combination-sum-iv)
* [2466. Count Ways To Build Good Strings](#2466-count-ways-to-build-good-strings) 1694
* [2266. Count Number of Texts](#2266-count-number-of-texts) 1857
* [2533. Number of Good Binary Strings](#2533-number-of-good-binary-strings) 1694

### 1.2 House Robber

* [198. House Robber](#198-house-robber)
* [740. Delete and Earn](#740-delete-and-earn)
* [2320. Count Number of Ways to Place Houses](#2320-count-number-of-ways-to-place-houses) 1608
* [213. House Robber II](#213-house-robber-ii)

### 1.3 aximum Subarray

* [53. Maximum Subarray]()

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
            cost[i] += min(cost[i - 1], cost[i - 2])
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
            return sum(dfs(i + t) for i in nums)
        return dfs(0)
```

### 2466. Count Ways To Build Good Strings

- same as 2466 (unbounded knapsack)

```python
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(n):
            if n < 0:
                return 0
            if n == 0:
                return 1
            return sum(dfs(n - num) % mod for num in [zero, one])
        return sum(dfs(x) for x in range(low, high + 1)) % mod 
```

### 2266. Count Number of Texts

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
        
        @cache
        def dfs(c, count):
            if count <= 1:
                return 1
            elif count == 2:
                return 2
            elif count == 3:
                return 4
            elif c in '234568':
                return dfs(c, count - 1) + dfs(c, count - 2) + dfs(c, count - 3) % mod
            else:
                return dfs(c, count - 1) + dfs(c, count - 2) + dfs(c, count - 3) + dfs(c, count - 4) % mod
        res = 1
        for c, count in arr:
            res *= dfs(c, count) % mod
            dfs.cache_clear()
        return res % mod
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
        def dfs(n):
            if n < 0:
                return 0
            if n == 0:
                return 1
            return sum(dfs(n - num) % mod for num in [zeroGroup, oneGroup]) % mod
        return sum(dfs(x) for x in range(minLength, maxLength + 1)) % mod 
```

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