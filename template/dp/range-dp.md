## template

### 516. Longest Palindromic Subsequence

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        @cache
        def dfs(i, j):
            if i == j:
                return 1
            if i > j:
                return 0
            if s[i] != s[j]:
                return max(dfs(i + 1, j), dfs(i, j - 1))
            else:
                return dfs(i + 1, j - 1) + 2
        return dfs(0, len(s) - 1)
```

### 1039. Minimum Score Triangulation of Polygon

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        v = values
        n = len(v)
        @cache
        def dfs(i, j):
            if i + 1 == j:
                return 0
            res = inf
            for k in range(i + 1, j):
                res = min(res, dfs(i, k) + dfs(k, j) + v[i] * v[j] * v[k])
            return res
        return dfs(0, n - 1)
```

### 1547. Minimum Cost to Cut a Stick

```python
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        @cache
        def dfs(l, r):
            res = inf
            for c in cuts:
                if l < c < r:
                    res = min(res, dfs(l, c) + dfs(c, r) + r - l)
            return res if res != inf else 0
        return dfs(0, n)
```

### 312. Burst Balloons

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        @cache
        def dfs(l, r):
            if l > r:
                return 0
            res = 0
            for i in range(l, r + 1):
                res = max(res, nums[l - 1] * nums[i] * nums[r + 1] + dfs(l, i - 1) + dfs(i + 1, r))
            return res
        return dfs(1, len(nums) - 2)
```

### 1312. Minimum Insertion Steps to Make a String Palindrome

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        @cache
        def dfs(i, j):
            if i >= j:
                return 0
            if s[i] == s[j]:
                return dfs(i + 1, j - 1)
            else:
                return min(dfs(i + 1, j), dfs(i, j - 1)) + 1
        return dfs(0, n - 1)
```

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        def LCS(text1, text2):
            R, C = len(text1) + 1, len(text2) + 1
            dp = [[0] * C for i in range(R)]
            for i in range(1, R):
                for j in range(1, C):
                    if text1[i - 1] == text2[j - 1]:
                        dp[i][j] = 1 + dp[i - 1][j - 1]
                    else:
                        dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
            return dp[-1][-1]
        return len(s) - LCS(s, s[::-1])
```

### 1000. Minimum Cost to Merge Stones (not understand totally)

```python
class Solution:
    def mergeStones(self, stones: List[int], k: int) -> int:
        n = len(stones)
        if (n - 1) % (k - 1):
            return -1
        s = list(accumulate(stones, initial=0))
        @cache  
        def dfs(i: int, j: int, p: int) -> int:
            if p == 1:
                print(i, j, k, s[j + 1] - s[i])
                return 0 if i == j else dfs(i, j, k) + s[j + 1] - s[i]
            return min(dfs(i, m, 1) + dfs(m + 1, j, p - 1) for m in range(i, j, k - 1))
        return dfs(0, n - 1, 1)
```

### 375. Guess Number Higher or Lower II

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        @cache
        def dfs(i, j):
            if i >= j:
                return 0
            res = inf
            for k in range(i, j + 1):
                res = min(res, max(dfs(i, k - 1), dfs(k + 1, j)) + k)
            return res
        return dfs(1, n)
```

### 3041. Maximize Consecutive Elements in an Array After Modification

```python
class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        d = defaultdict(int)
        nums.sort()
        for x in nums:
            d[x + 1] = d[x] + 1
            d[x] = d[x - 1] + 1
        return max(d.values())
```

## range dp

- 3 loops of i, j, k
- dp[i][j] from 1 to i numbers choose j split and get the result
- dp[i][j] = max(dp[i][j], dp[k - 1][j - 1] + from j to i)

### 813. Largest Sum of Averages

```python
class Solution:
    def largestSumOfAverages(self, nums: List[int], m: int) -> float:
        n = len(nums)
        nums = list(accumulate(nums, initial = 0))
        dp = [[0] * (m + 1) for c in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                if j == 1:
                    dp[i][j] = nums[i] / i 
                else:
                    for k in range(2, i + 1):
                        dp[i][j] = max(dp[i][j], dp[k - 1][j - 1] + (nums[i] - nums[k - 1]) / (i - k + 1))
        return dp[n][m]
```

### 410. Split Array Largest Sum

```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        n = len(nums)
        nums = list(accumulate(nums, initial = 0))
        dp = [[inf] * (m + 1) for c in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                if j == 1:
                    dp[i][j] = nums[i]
                else:
                    for k in range(2, i + 1):
                        t = dp[k - 1][j - 1]
                        if t < nums[i] - nums[k - 1]:
                            t = nums[i] - nums[k - 1]
                        if t < dp[i][j]:
                            dp[i][j] = t
                        # dp[i][j] = min(dp[i][j], max(dp[k - 1][j - 1], (nums[i] - nums[k - 1])))
        return dp[n][m]
```

### 664. Strange Printer

```python
class Solution:
    def strangePrinter(self, s: str) -> int:
        @cache
        def dfs(i, j):
            if i == j:
                return 1
            if s[i] == s[j]:
                return dfs(i, j - 1)
            return min(dfs(i, k) + dfs(k + 1, j) for k in range(i, j))
        n = len(s)
        return dfs(0, n - 1)
```

### 87. Scramble String

```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        @cache
        def dfs(s1, s2):
            if s1 == s2:
                return True
            L = len(s1)
            for i in range(1, L):
                if dfs(s1[:i], s2[:i]) and dfs(s1[i:], s2[i:]):
                    return True
                if dfs(s1[:i], s2[L-i:]) and dfs(s1[i:], s2[:L-i]):
                    return True
            return False
        return dfs(s1, s2)
```