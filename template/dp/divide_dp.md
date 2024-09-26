## 1. Can partition

* [2369. Check if There is a Valid Partition For The Array](#2369-check-if-there-is-a-valid-partition-for-the-array)
* [139. Word Break](#139-word-break)

## 2. Optimize partition

* [132. Palindrome Partitioning II](#132-palindrome-partitioning-ii)

## 2. Best partition

### 2369. Check if There is a Valid Partition For The Array

```python
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        @cache
        def dfs(i):
            if i == n:
                return True
            res1, res2 = False, False
            if i + 1 < n and nums[i] == nums[i + 1]:
                res1 = dfs(i + 2)
            if i + 2 < n and (nums[i] == nums[i + 1] == nums[i + 2] or (nums[i] + 1 == nums[i + 1] and nums[i + 1] + 1 == nums[i + 2])):
                res2 = dfs(i + 3)
            return res1 or res2
        return dfs(0)
```

### 139. Word Break

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)
        n = len(s)
        @cache 
        def dfs(i):
            if i == n:
                return True 
            res = False
            for j in range(n):
                if s[i: j + 1] in wordDict:
                    res = res or dfs(j + 1)
            return res 
        return dfs(0)
```

- based on leetcode 5

### 132. Palindrome Partitioning II

```python
class Solution:
    def minCut(self, s: str) -> int:
        @cache
        def dfs(i):
            if i == n:
                return 0
            res = inf 
            for j in range(i, n):
                if dp[i][j]:
                    res = min(res, dfs(j + 1) + 1)
            return res

        n = len(s)
        dp = [[False] * n for r in range(n)]
        for j in range(n):
            for i in range(j + 1):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
        return dfs(0) - 1
```

- interval dp

### 5. Longest Palindromic Substring

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n, res = len(s), ''
        dp = [[False] * n for r in range(n)]
        for j in range(n):
            for i in range(j + 1):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    res = max(res, s[i: j + 1], key = len)
        return res 
```