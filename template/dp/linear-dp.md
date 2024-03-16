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
