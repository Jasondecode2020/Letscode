## 0-1 knapsack - template

```python
@cache
def f(v, w, t, i):
    if i == len(v):
        return 0
    elif t < 0:
        return -inf 
    return max(f(v, w, t, i + 1), f(v, w, t - w[i], i + 1) + v[i])
```

## unbounded knapsack - template

```python
@cache
def f(v, w, t, i):
    if i == len(v):
        return 0
    elif t < 0:
        return -inf 
    # if choose: (t - w[i], i) i same because can choose again
    # if not choose: (t, i + 1), i + 1 means go to the next
    return max(f(v, w, t, i + 1), f(v, w, t - w[i], i) + v[i]) 
```


## 3 Knapsack (18)

### 1 0-1 knapsack

* [2915. Length of the Longest Subsequence That Sums to Target](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [416. Partition Equal Subset Sum](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [494. Target Sum](#494-Target-Sum)
* [2787. Ways to Express an Integer as Sum of Powers](#2787-ways-to-express-an-integer-as-sum-of-powers)
* [3180. Maximum Total Reward Using Operations I](#3180-maximum-total-reward-using-operations-i)
* [474. Ones and Zeroes](#474-ones-and-zeroes)
* [1049. Last Stone Weight II](#1049-last-stone-weight-ii)
* [1774. Closest Dessert Cost](#1774-closest-dessert-cost)
* [879. Profitable Schemes](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [805. Split Array With Same Average](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [923. 3Sum With Multiplicity](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [2291. Maximum Profit From Trading Stocks](#2291-maximum-profit-from-trading-stocks)
* [3082. Find the Sum of the Power of All Subsequences](#3082-find-the-sum-of-the-power-of-all-subsequences)

### 2 unbounded knapsack

* [322. Coin Change](#322-coin-change)
* [518. Coin Change II](#518-coin-change-ii)
* [279. Perfect Squares](#279-perfect-squares)
* [377. Combination Sum IV](#377-combination-sum-iv-unbounded-knapsack-with-sequence)
* [1449. Form Largest Integer With Digits That Add up to Target](#1449-form-largest-integer-with-digits-that-add-up-to-target)
* [3183. The Number of Ways to Make the Sum](#3183-the-number-of-ways-to-make-the-sum)

### 3 mutiple knapsack

### 4 group knapsack

* [1155. Number of Dice Rolls With Target Sum](#1155-number-of-dice-rolls-with-target-sum)
* [1981. Minimize the Difference Between Target and Chosen Elements](#1981-minimize-the-difference-between-target-and-chosen-elements)
* [2218. Maximum Value of K Coins From Piles](#2218-maximum-value-of-k-coins-from-piles)

### 2915. Length of the Longest Subsequence That Sums to Target

```python
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        nums.sort(reverse = True)
        N = len(nums)
        @cache
        def f(t, i):
            if t > target:
                return -inf
            if i == N:
                return 0 if t == target else -inf 
            return max(f(t, i + 1), f(t + nums[i], i + 1) + 1)
        res = f(0, 0)
        f.cache_clear()
        return res if res != -inf else -1
```


### 416. Partition Equal Subset Sum

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        total = sum(nums)
        if total % 2:
            return False 
        total //= 2
        @cache
        def dfs(i, t):
            if i == n:
                return True if t == total else False 
            return dfs(i + 1, t) or dfs(i + 1, t + nums[i])
        return dfs(0, 0)
```

### 494. Target Sum

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        n = len(nums)
        @cache
        def dfs(i, t):
            if i == n:
                return 1 if t == target else 0
            return dfs(i + 1, t - nums[i]) + dfs(i + 1, t + nums[i])
        return dfs(0, 0)
```

### 2787. Ways to Express an Integer as Sum of Powers

```python
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        nums = list(range(1, n + 1))[::-1]
        N = len(nums)
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, t):
            if t > n:
                return 0 
            if i == n:
                return 1 if t == n else 0
            return (dfs(i + 1, t) + dfs(i + 1, t + nums[i] ** x)) % mod 
        return dfs(0, 0)
```

### 3180. Maximum Total Reward Using Operations I

```python
rewardValues.sort()
        n = len(rewardValues)
        @cache
        def dfs(i, t):
            if i == n:
                return t
            res = dfs(i + 1, t)
            if rewardValues[i] > t:
                res = max(res, dfs(i + 1, t + rewardValues[i]))
            return res
        res = dfs(0, 0)
        dfs.cache_clear()
        return res
```

### 474. Ones and Zeroes (multi knapsacks)

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        @cache
        def f(a, b, i):
            if i == len(strs):
                return 0 if (a <= m and b <= n) else -inf
            if a > m or b > n: # optimize
                return -inf
            ones, zeros = strs[i].count('1'), strs[i].count('0')
            return max(f(a, b, i + 1), f(a + zeros, b + ones, i + 1) + 1)
        res = f(0, 0, 0)
        return res if res != -inf else 0
```

### 1049. Last Stone Weight II

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        mx = max(stones)
        @cache
        def f(t, i):
            if i == len(stones):
                return abs(t)
            if t > mx: # optimize
                return inf
            return min(f(t - stones[i], i + 1), f(t + stones[i], i + 1))
        return f(0, 0)
```

### 2291. Maximum Profit From Trading Stocks

```python
class Solution:
    def maximumProfit(self, present: List[int], future: List[int], budget: int) -> int:
        @cache
        def dfs(i, b):
            if b > budget:
                return -inf 
            if i == n:
                return 0 if b <= budget else -inf 
            res = dfs(i + 1, b)
            if future[i] > present[i]:
                res = max(res, dfs(i + 1, b + present[i]) + future[i] - present[i])
            return res 

        n = len(present)
        res = dfs(0, 0)
        dfs.cache_clear()
        return res if res != -inf else 0
```

### 3082. Find the Sum of the Power of All Subsequences

```python
class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        @cache
        def dfs(i, t, selected):
            if t > k:
                return 0
            if i == n:
                return 2 ** (n - selected) if t == k else 0 
            return dfs(i + 1, t, selected) + dfs(i + 1, t + nums[i], selected + 1)
        
        n = len(nums)
        mod = 10 ** 9 + 7
        res = dfs(0, 0, 0)
        dfs.cache_clear()
        return res % mod
```

### 1774. Closest Dessert Cost

```python
class Solution:
    def closestCost(self, baseCosts: List[int], toppingCosts: List[int], target: int) -> int:
        def dfs(i, cost):
            if i == len(toppingCosts):
                if abs(cost - target) < abs(self.res - target) or (abs(cost - target) == abs(self.res - target) and cost < target):
                    self.res = cost 
                return

            dfs(i + 1, cost)
            dfs(i + 1, cost + toppingCosts[i])
            dfs(i + 1, cost + toppingCosts[i] * 2)
        
        self.res = inf 
        for base in baseCosts:
            dfs(0, base)
        return self.res 
```

### 879. Profitable Schemes (multi dimentional)

```python
mod = 10 ** 9 + 7

class Solution:
    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        res = []
        for g, p in zip(group, profit):
            res.append(((g / p) if p != 0 else inf, g, p))
        res.sort(reverse = True)
        group, profit = [item[1] for item in res], [item[2] for item in res]
        N = len(group)
        @cache
        def f(p, g, i):
            if g > n:
                return 0
            if i == N:
                return 1 if p >= minProfit else 0
            return f(p, g, i + 1) + f(p + profit[i], g + group[i], i + 1)
        return f(0, 0, 0) % mod
```



### 805. Split Array With Same Average

```python
class Solution:
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        nums.sort()
        total, N = sum(nums), len(nums)
        average = total / N
        # total / N = t / n, t is integer
        if all(total * i % N != 0 for i in range(1, N // 2 + 1)):
            return False
        @cache
        def f(t, n, i):
            if t > n * average:
                return False
            if n and n < N and t / n == average:
                return True
            if i == N:
                if n == 0 or n == N:
                    return False
                return True if t * N == total * n else False
            return f(t, n, i + 1) or f(t + nums[i], n + 1, i + 1)
        return f(0, 0, 0)
```


### 923. 3Sum With Multiplicity

```python
class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        arr.sort(reverse = True)
        n = len(arr)
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, total, count):
            if i >= n:
                return 1 if total == target and count == 3 else 0
            if total > target or count > 3:
                return 0
            return dfs(i + 1, total, count) + dfs(i + 1, total + arr[i], count + 1)
        return dfs(0, 0, 0) % mod

```

### 279. Perfect Squares

```python
square = [i * i for i in range(100, 0, -1)]
class Solution:
    def numSquares(self, n: int) -> int:
        @cache
        def f(t, i):
            if t > n:
                return inf 
            if i == len(square):
                return 0 if t == n else inf
            return min(f(t, i + 1), f(t + square[i], i) + 1) 
        res = f(0, 0)
        f.cache_clear()
        return res
```

### 322. Coin Change

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse = True)
        @cache
        def f(t, i):
            if t > amount:
                return inf 
            if i == len(coins):
                return 0 if t == amount else inf 
            return min(f(t, i + 1), f(t + coins[i], i) + 1) 
        res = f(0, 0)
        return res if res != inf else -1
```

### 518. Coin Change II

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        coins.sort(reverse = True)
        N = len(coins)
        @cache
        def f(t, i):
            if t > amount:
                return 0
            if i == N:
                return 1 if t == amount else 0
            return f(t, i + 1) + f(t + coins[i], i)
        res = f(0, 0)
        return res if res != inf else -1
```

### 377. Combination Sum IV (unbounded knapsack with sequence)

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @cache
        def f(t):
            if t > target:
                return 0
            if t == target:
                return 1
            return sum(f(t + n) for n in nums)
        return f(0)
```

### 1449. Form Largest Integer With Digits That Add up to Target

```python
class Solution:
    def largestNumber(self, cost: List[int], target: int) -> str:
        @cache
        def dfs(t):
            if t == 0:
                return ""
            ans = "0"
            for c in cost:
                if t >= c:
                    res = dfs(t - c)
                    if res != '0':
                        if len(res) + 1 > len(ans):
                            ans = d[c] + res
                        elif len(res) + 1 == len(ans):
                            ans = max(ans, d[c] + res, key = int)
            return ans

        d = dict((c, str(i)) for i, c in enumerate(cost, 1))
        cost = sorted(d.keys())
        return dfs(target)
```

### 3183. The Number of Ways to Make the Sum

```python
class Solution:
    def numberOfWays(self, n: int) -> int:
        mod = 10**9 + 7
        coins = [1, 2, 6]
        f = [0] * (n + 1)
        f[0] = 1
        for x in coins:
            for j in range(x, n + 1):
                f[j] = (f[j] + f[j - x]) % mod
        ans = f[n]
        if n >= 4:
            ans = (ans + f[n - 4]) % mod
        if n >= 8:
            ans = (ans + f[n - 8]) % mod
        return ans
```

### 2585. Number of Ways to Earn Points

```python
mod = 10 ** 9 + 7
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        n = len(types)
        @cache
        def f(i, t):
            if i < 0:
                return 1 if t == 0 else 0
            count, mark = types[i]
            res = 0
            for k in range(min(count, t // mark) + 1):
                res += f(i - 1, t - k * mark)
            return res
        return f(n - 1, target) % mod
```

### 2915. Length of the Longest Subsequence That Sums to Target

```python
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        nums.sort()
        N = len(nums)
        @cache
        def f(t, i):
            if t > target:
                return -inf 
            if i == N:
                return 0 if t == target else -inf 
            return max(f(t, i + 1), f(t + nums[i], i + 1) + 1)
        res = f(0, 0)
        f.cache_clear()
        return res if res != -inf else -1
```

### 416. Partition Equal Subset Sum

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 == 1:
            return False
        half = sum(nums) // 2
        @cache
        def f(t, i):
            if i == len(nums):
                return True if t == half else False
            return f(t, i + 1) or f(t + nums[i], i + 1)
        return f(0, 0)
```

### 494. Target Sum

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        @cache
        def f(t, i):
            if i == len(nums):
                return 1 if t == target else 0
            return f(t - nums[i], i + 1) + f(t + nums[i], i + 1)
        return f(0, 0)
```


### 474. Ones and Zeroes

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        @cache
        def f(a, b, i):
            if i == len(strs):
                return 0 if (a <= m and b <= n) else -inf
            if a > m or b > n:
                return -inf
            ones, zeros = strs[i].count('1'), strs[i].count('0')
            return max(f(a, b, i + 1), f(a + zeros, b + ones, i + 1) + 1)
        res = f(0, 0, 0)
        return res if res != -inf else 0
```



### 1049. Last Stone Weight II

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        mx = max(stones)
        @cache
        def f(t, i):
            if i == len(stones):
                return abs(t)
            if t > mx:  # optimize
                return inf
            return min(f(t - stones[i], i + 1), f(t + stones[i], i + 1))

        return f(0, 0)
```

### 279. Perfect Squares

```python
square = [i * i for i in range(100, 0, -1)]
class Solution:
    def numSquares(self, n: int) -> int:
        @cache
        def f(t, i):
            if t > n:
                return inf 
            if i == len(square):
                return 0 if t == n else inf
            return min(f(t, i + 1), f(t + square[i], i) + 1) 
        res = f(0, 0)
        f.cache_clear()
        return res
```

### 1449. Form Largest Integer With Digits That Add up to Target

### 2585. Number of Ways to Earn Points

```python
mod = 10 ** 9 + 7
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        n = len(types)
        @cache
        def f(i, t):
            if i < 0:
                return 1 if t == 0 else 0
            count, mark = types[i]
            res = 0
            for k in range(min(count, t // mark) + 1):
                res += f(i - 1, t - k * mark)
            return res
        return f(n - 1, target) % mod
```

### 1155. Number of Dice Rolls With Target Sum

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, t):
            if t > target:
                return 0
            if i == n:
                return 1 if t == target else 0
            res = 0
            for j in range(1, k + 1):
                res += dfs(i + 1, t + j)
            return res % mod
        return dfs(0, 0) % mod
```

### 1981. Minimize the Difference Between Target and Chosen Elements

```python
class Solution:
    def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
        # R, C = len(mat), len(mat[0])
        # self.res = inf 
        # @cache
        # def dfs(r, t):
        #     if r == R:
        #         self.res = min(self.res, abs(t - target))
        #         return
        #     for c in range(C):
        #         dfs(r + 1, t + mat[r][c])
        # dfs(0, 0)
        # return self.res
        @cache
        def f(i):
            if i == 0:
                return set(mat[0])
            return set(x + y for x in mat[i] for y in f(i - 1))
        R = len(mat)
        return min(abs(target - v) for v in f(R - 1))
```

### 2218. Maximum Value of K Coins From Piles

```python
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        n = len(piles)
        @cache
        def f(i, k):
            if i == n or k == 0:
                return 0
            res = f(i + 1, k)
            pre = 0
            for j in range(min(k, len(piles[i]))):
                pre += piles[i][j]
                res = max(res, f(i + 1, k - j - 1) + pre)
            return res 
        return f(0, k)
```