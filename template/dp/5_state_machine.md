
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
        res, lowest = 0, prices[0]
        for p in prices:
            lowest = min(lowest, p)
            res = max(res, p - lowest)
        return res 
```

### 122. Best Time to Buy and Sell Stock II

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        res = 0
        for i in range(1, n):
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

### 1363. Largest Multiple of Three

- greedy

```python
class Solution:
    def largestMultipleOfThree(self, digits: List[int]) -> str:
        d = defaultdict(list)
        for n in digits:
            d[n % 3].append(n)
        if (len(d[1]) + 2 * len(d[2])) % 3 == 1:
            if len(d[1]) >= 1:
                num = sorted(d[1])[0]
                d[1].remove(num)
            elif len(d[2]) >= 2:
                nums = sorted(d[2])[:2]
                for num in nums:
                    d[2].remove(num)
            else:
                return ''
        if (len(d[1]) + 2 * len(d[2])) % 3 == 2:
            if len(d[2]) >= 1:
                num = sorted(d[2])[0]
                d[2].remove(num)
            elif len(d[1]) >= 2:
                nums = sorted(d[1])[:2]
                for num in nums:
                    d[1].remove(num)
            else:
                return ''
        arr = d[0] + d[1] + d[2]
        arr.sort(reverse = True)
        if arr and arr[0] == 0:
            return '0'
        s = [str(i) for i in arr]
        res = ''.join(s)
        return res
```

### 1186. Maximum Subarray Sum with One Deletion

### 801. Minimum Swaps To Make Sequences Increasing

```python
class Solution:
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        f = [[inf, inf] for i in range(n)]
        f[0] = [0, 1]
        for i in range(1, n):
            if nums1[i - 1] < nums1[i] and nums2[i - 1] < nums2[i]:
                f[i][0] = f[i - 1][0]
                f[i][1] = f[i - 1][1] + 1
            if nums2[i - 1] < nums1[i] and nums1[i - 1] < nums2[i]:
                f[i][0] = min(f[i][0], f[i - 1][1])
                f[i][1] = min(f[i][1], f[i - 1][0] + 1)
        return min(f[-1])
```

### 2036. Maximum Alternating Subarray Sum

```python
class Solution:
    def maximumAlternatingSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        f = [[-inf, -inf] for i in range(n)]
        f[0] = [-inf, nums[0]]
        res = nums[0]
        for i in range(1, n):
            f[i][0] = f[i - 1][1] - nums[i]
            f[i][1] = max(f[i - 1][0] + nums[i], nums[i])
            res = max(res, max(f[i]))
        return res 
```

### 1911. Maximum Alternating Subsequence Sum

```python
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        even, odd = nums[0], 0
        for i in range(1, len(nums)):
            even, odd = max(even, odd + nums[i]), max(odd, even - nums[i])
        return even 
```