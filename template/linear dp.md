* 70. Climbing Stairs
* 509. Fibonacci Number
* 1137. N-th Tribonacci Number
* 746. Min Cost Climbing Stairs

### 70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        first, second = 1, 2
        for i in range(3, n + 1):
            first, second = second, first + second 
        return second if n >= 2 else 1
```

### 509. Fibonacci Number

```python
class Solution:
    def fib(self, n: int) -> int:
        first, second = 0, 1
        for i in range(2, n + 1):
            first, second = second, first + second 
        return second if n >= 1 else 0
```

### 1137. N-th Tribonacci Number

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        first, second, third = 0, 1, 1
        for i in range(3, n + 1):
            first, second, third = second, third, first + second + third
        return third if n >= 1 else 0
```

### 746. Min Cost Climbing Stairs

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        for i in range(2, len(cost)):
            cost[i] = min(cost[i] + cost[i - 1], cost[i] + cost[i - 2])
        return min(cost[-1], cost[-2])
```

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

### 1025. Divisor Game

```python
class Solution:
    def divisorGame(self, n: int) -> bool:
        @cache
        def dfs(n):
            if n == 1:
                return False

            for i in range(1, n):
                if n % i == 0:
                    if not dfs(n - i):
                        return True
            return False
        return dfs(n)
```

### 118. Pascal's Triangle

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for i in range(2, numRows + 1):
            lastRow = res[-1]
            lastRow = [0] + lastRow + [0]
            lastRow = [lastRow[i] + lastRow[i + 1] for i in range(len(lastRow) - 1)]
            res.append(lastRow)
        return res
```

### 119. Pascal's Triangle II

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [[1]]
        for i in range(1, rowIndex + 1):
            lastRow = res[-1]
            lastRow = [0] + lastRow + [0]
            lastRow = [lastRow[i] + lastRow[i + 1] for i in range(len(lastRow) - 1)]
            res.append(lastRow)
        return res[-1]
```

### 338. Counting Bits

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        res = [[1]]
        for i in range(1, rowIndex + 1):
            lastRow = res[-1]
            lastRow = [0] + lastRow + [0]
            lastRow = [lastRow[i] + lastRow[i + 1] for i in range(len(lastRow) - 1)]
            res.append(lastRow)
        return res[-1]
```

### 2826. Sorting Three Groups

```python
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * 3 for r in range(n + 1)]
        for i in range(1, n + 1):
            if nums[i - 1] == 1:
                dp[i][0] = dp[i - 1][0]
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + 1
                dp[i][2] = min(dp[i - 1][0], dp[i - 1][1], dp[i - 1][2]) + 1
            elif nums[i - 1] == 2:
                dp[i][0] = dp[i - 1][0] + 1
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1])
                dp[i][2] = min(dp[i - 1][0], dp[i - 1][1], dp[i - 1][2]) + 1
            else:
                dp[i][0] = dp[i - 1][0] + 1
                dp[i][1] = min(dp[i - 1][0], dp[i - 1][1]) + 1
                dp[i][2] = min(dp[i - 1][0], dp[i - 1][1], dp[i - 1][2])
        return min(dp[-1])
```