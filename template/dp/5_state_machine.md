
## 5 State Machine (18)

* [121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)
* [122. Best Time to Buy and Sell Stock II](#122-best-time-to-buy-and-sell-stock-ii)
* [123. Best Time to Buy and Sell Stock III](#123-best-time-to-buy-and-sell-stock-iii)
* [188. Best Time to Buy and Sell Stock IV](#188-best-time-to-buy-and-sell-stock-iv)
* [309. Best Time to Buy and Sell Stock with Cooldown](#309-best-time-to-buy-and-sell-stock-with-cooldown)
* [714. Best Time to Buy and Sell Stock with Transaction Fee](#714-best-time-to-buy-and-sell-stock-with-transaction-fee)

* [1493. Longest Subarray of 1's After Deleting One Element](#1493-longest-subarray-of-1s-after-deleting-one-element)
* [1395. Count Number of Teams](#1395-count-number-of-teams)

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

### 3259. Maximum Energy Boost From Two Drinks

```python
class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        @cache
        def f(i, A):
            if i >= n:
                return 0
            res = 0
            if A:
                res = max(f(i + 1, True), f(i + 2, False)) + energyDrinkA[i]
            else:
                res = max(f(i + 2, True), f(i + 1, False)) + energyDrinkB[i]
            return res 
        n = len(energyDrinkA)
        return max(f(0, True), f(0, False))
```

### 2745. Construct the Longest New String

```python
class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        res = 0
        if y > x:
            res = (x + x + 1 + z) * 2
        elif y < x:
            res = (y + y + 1 + z) * 2 
        else:
            res = (x + y + z) * 2
        return res
```

### 2222. Number of Ways to Select Buildings

```python
class Solution:
    def numberOfWays(self, s: str) -> int:
        zero, one = s.count('0'), s.count('1')
        def check(b, total):
            left, right, res = 0, 0, 0
            for i, c in enumerate(s):
                if c == b:
                    right = total - left
                    res += left * right
                else:
                    left += 1
            return res 
        return check('0', one) + check('1', zero)
```

### 3290. Maximum Multiplication Score

```python
class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        @cache
        def dfs(i, N):
            if i == n:
                return 0 if N == 4 else -inf
            res = dfs(i + 1, N)
            if N < 4:
                res = max(res, dfs(i + 1, N + 1) + b[i] * a[N])
            return res
            
        n = len(b)
        return dfs(0, 0)
```