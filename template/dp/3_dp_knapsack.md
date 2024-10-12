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

### 3.1 0-1 (8)

* [62. Unique Paths](#62-unique-paths)
* [63. Unique Paths II](#63-unique-paths-ii)
* [64. Minimum Path Sum](#64-minimum-path-sum)
* [120. Triangle](#120-triangle)
* [931. Minimum Falling Path Sum](#931-minimum-falling-path-sum)
* [2684. Maximum Number of Moves in a Grid](#2684-maximum-number-of-moves-in-a-grid)
* [1289. Minimum Falling Path Sum II](#1289-minimum-falling-path-sum-ii)
* [2304. Minimum Path Cost in a Grid](#2304-minimum-path-cost-in-a-grid)

### 3.2 unbounded (10)

* [1594. Maximum Non Negative Product in a Matrix](#62-unique-paths)
* [2435. Paths in Matrix Whose Sum Is Divisible by K](#2435-paths-in-matrix-whose-sum-is-divisible-by-k)
* [174. Dungeon Game](#174-dungeon-game)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2267. Check if There Is a Valid Parentheses String Path](#2267-check-if-there-is-a-valid-parentheses-string-path)
* [2328. Number of Increasing Paths in a Grid](#2328-number-of-increasing-paths-in-a-grid)
* [2510. Check if There is a Path With Equal Number of 0's And 1's](#2510-check-if-there-is-a-path-with-equal-number-of-0s-and-1s)
* [1463. Cherry Pickup II](#1463-cherry-pickup-ii)
* [741. Cherry Pickup](#741-cherry-pickup)
* [1937. Maximum Number of Points with Cost](#1937-maximum-number-of-points-with-cost)

### 3.3 mutiple (1)

* [1594. Maximum Non Negative Product in a Matrix](#62-unique-paths)

### 3.2 group (3)

* [1155. Number of Dice Rolls With Target Sum](#1155-number-of-dice-rolls-with-target-sum)
* [1981. Minimize the Difference Between Target and Chosen Elements](#2435-paths-in-matrix-whose-sum-is-divisible-by-k)
* [2218. Maximum Value of K Coins From Piles](#174-dungeon-game)


### 0-1 knapsack question list

* [494. Target Sum](#494-Target-Sum)
* [416. Partition Equal Subset Sum](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [474. Ones and Zeroes](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [1049. Last Stone Weight II](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [879. Profitable Schemes](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [2915. Length of the Longest Subsequence That Sums to Target](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [805. Split Array With Same Average](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [2787. Ways to Express an Integer as Sum of Powers](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)
* [923. 3Sum With Multiplicity](#2915-Length-of-the-Longest-Subsequence-That-Sums-to-Target)

### unbounded knapsack question list

* 279. Perfect Squares
* 322. Coin Change
* 518. Coin Change II
* 377. Combination Sum IV (unbounded knapsack with sequence)

## multi knapsack

### ###################################################################### 0-1 knapsack

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

### 2787. Ways to Express an Integer as Sum of Powers

```python
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

### ###################################################################### unbounded knapsack

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

### 2787. Ways to Express an Integer as Sum of Powers

```python
mod = 10 ** 9 + 7
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        nums = [i for i in range(n, 0, -1)]
        N = len(nums)
        @cache
        def f(t, i):
            if t > n:
                return 0
            if i == N:
                return 1 if t == n else 0
            return (f(t, i + 1) + f(t + nums[i] ** x, i + 1)) % mod
        res = f(0, 0)
        return res
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

1774. 最接近目标价格的甜点成本
879. 盈利计划 2204
3082. 求出所有子序列的能量和 2242
956. 最高的广告牌 2381
2518. 好分区的数目 2415
2742. 给墙壁刷油漆 2425
LCP 47. 入场安检
2291. 最大股票收益（会员题）
2431. 最大限度地提高购买水果的口味（会员题）

# ###################################################################################


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

### 1449. Form Largest Integer With Digits That Add up to Target

###########################################################################################

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

#####################################################################################################

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

### 2218. Maximum Value of K Coins From Piles

## 5 State Machine (18)

* [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
* [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
* [123. Best Time to Buy and Sell Stock III](#123-best-time-to-buy-and-sell-stock-iii)
* [188. Best Time to Buy and Sell Stock IV](#188-best-time-to-buy-and-sell-stock-iv)
* [309. Best Time to Buy and Sell Stock with Cooldown](#309-best-time-to-buy-and-sell-stock-with-cooldown)
* [714. Best Time to Buy and Sell Stock with Transaction Fee](#714-best-time-to-buy-and-sell-stock-with-transaction-fee)

### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowest, profit = prices[0], 0
        for price in prices:
            lowest = min(lowest, price)
            profit = max(profit, price - lowest)
        return profit
```

### 122. Best Time to Buy and Sell Stock II

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # [7,1,5,3,6,4]
        res = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                res += prices[i] - prices[i - 1]
        return res
```

### 123. Best Time to Buy and Sell Stock III

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding, count):
            if i >= n or count == 2:
                return 0
            cooldown = dfs(i + 1, holding, count)
            if holding:
                sell = dfs(i + 1, False, count + 1) + prices[i]
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True, count) - prices[i]
                res = max(cooldown, buy)
            return res 
        return dfs(0, False, 0)
```

### 188. Best Time to Buy and Sell Stock IV

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding, count):
            if i >= n or count == k:
                return 0
            cooldown = dfs(i + 1, holding, count)
            if holding:
                sell = dfs(i + 1, False, count + 1) + prices[i]
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True, count) - prices[i]
                res = max(cooldown, buy)
            return res 
        return dfs(0, False, 0)
```

### 309. Best Time to Buy and Sell Stock with Cooldown

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding):
            if i >= n:
                return 0
            cooldown = dfs(i + 1, holding)
            if holding:
                sell = dfs(i + 2, False) + prices[i]
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True) - prices[i]
                res = max(cooldown, buy)
            return res
        return dfs(0, False)
```

### 714. Best Time to Buy and Sell Stock with Transaction Fee

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        @cache
        def dfs(i, holding):
            if i >= n:
                return 0
            cooldown = dfs(i + 1, holding)
            if holding:
                sell = dfs(i + 1, False) + prices[i] - fee 
                res = max(cooldown, sell)
            else:
                buy = dfs(i + 1, True) - prices[i]
                res = max(cooldown, buy)
            return res 
        return dfs(0, False)
```

### 1493. Longest Subarray of 1's After Deleting One Element

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        l, res = 0, 0
        zero = 0
        for r, n in enumerate(nums):
            if n == 0:
                zero += 1
            while zero > 1:
                if nums[l] == 0:
                    zero -= 1
                l += 1
            res = max(res, r - l)
        return res 
```

### 1395. Count Number of Teams

```python
class Solution:
    def numTeams(self, rating: List[int]) -> int:
        res = 0
        n = len(rating)
        for i in range(1, n - 1):
            left = sum(rating[j] < rating[i] for j in range(i))
            right = sum(rating[j] > rating[i] for j in range(i + 1, n))
            res += left * right
            left = sum(rating[j] > rating[i] for j in range(i))
            right = sum(rating[j] < rating[i] for j in range(i + 1, n))
            res += left * right
        return res
```