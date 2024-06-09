### 651. 4 Keys Keyboard

```python
class Solution:
    def maxA(self, n: int) -> int:
        dp = [i for i in range(n + 1)]
        for i in range(4, n + 1):
            for j in range(i - 2):
                dp[i] = max(dp[i], dp[j] * (i - j - 2 + 1))
        return dp[-1]
```

### 650. 2 Keys Keyboard

```python
class Solution:
    def minSteps(self, n: int) -> int:
        dp = [inf] * (n + 1)
        dp[1] = 0
        for i in range(2, n + 1):
            for j in range(2, i + 1):
                if i % j == 0:
                    k = i // j
                    dp[i] = min(dp[i], dp[k] + 1 + j - 1)
        return dp[-1]
```

### 2369. Check if There is a Valid Partition For The Array

```python
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        @cache
        def dfs(i):
            if i == n:
                return True  
            first, second = False, False          
            if i + 1 < n and nums[i] == nums[i + 1]:
                first = dfs(i + 2)
            if i + 2 < n and (nums[i] == nums[i + 1] == nums[i + 2] or (nums[i] + 1 == nums[i + 1] and nums[i + 1] + 1 == nums[i + 2])):
                second = dfs(i + 3) 
            return first or second 
        return dfs(0)
```

### 139. Word Break

- prefix idea: dp[i] means if dp[:i] is combined by the wordDict or not

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n, wordDict = len(s), set(wordDict)
        dp = [True] + [False] * n
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]
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