# template

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

- 1. dp[i][j]: the value between i to j inclusive
- 2. try to use i + 1, or j - 1 to make the problem smaller
- 3. find the boundary case: i == j or i > j

## Question list

## 1 Longest Panlindrome (7)

* [516. Longest Palindromic Subsequence](#516-longest-palindromic-subsequence)
* [1216. Valid Palindrome III](#1216-valid-palindrome-iii)
* [1312. Minimum Insertion Steps to Make a String Palindrome](#1312-minimum-insertion-steps-to-make-a-string-palindrome)
* [1771. Maximize Palindrome Length From Subsequences 2186](#1771-maximize-palindrome-length-from-subsequences)
* [1682. Longest Palindromic Subsequence II 2100](#1682-longest-palindromic-subsequence-ii)


* [1246. Palindrome Removal 2203](#1246-palindrome-removal)
* [730. Count Different Palindromic Subsequences](#730-count-different-palindromic-subsequences)

## 2 Others

* [5. Longest Palindromic Substring](#5-longest-palindromic-substring)
* [3040. Maximum Number of Operations With the Same Score II](#3040-maximum-number-of-operations-with-the-same-score-ii)
* [1770. Maximum Score from Performing Multiplication Operations](#1770-maximum-score-from-performing-multiplication-operations)
* [1039. Minimum Score Triangulation of Polygon](#1039-minimum-score-triangulation-of-polygon) vip
* [1130. Minimum Cost Tree From Leaf Values](#1130-minimum-cost-tree-from-leaf-values)

* [96. Unique Binary Search Trees](#96-unique-binary-search-trees)
* [375. Guess Number Higher or Lower II](#375-guess-number-higher-or-lower-ii)
* [1547. Minimum Cost to Cut a Stick](#1547-minimum-cost-to-cut-a-stick)
* [1000. Minimum Cost to Merge Stones (not understand totally)](#1000-minimum-cost-to-merge-stones-not-understand-totally)
* [87. Scramble String](#87-scramble-string)

* [312. Burst Balloons](#312-burst-balloons)
* [410. Split Array Largest Sum](#410-split-array-largest-sum)
* [664. Strange Printer](#664-strange-printer)
* [2464. Minimum Subarrays in a Valid Split](#2464-minimum-subarrays-in-a-valid-split)
* [813. Largest Sum of Averages](#813-largest-sum-of-averages)

* [2019. The Score of Students Solving Math Expression](TODO:)
* [471. Encode String with Shortest Length](TODO:)
* [3018. Maximum Number of Removal Queries That Can Be Processed I](TODO:)
* [546. Remove Boxes](TODO:)

### 516. Longest Palindromic Subsequence

- lcs

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        def lcs(s1, s2):
            R, C = len(s1) + 1, len(s2) + 1
            f = [[0] * C for i in range(R)]
            for i in range(1, R):
                for j in range(1, C):
                    if s1[i - 1] == s2[j - 1]:
                        f[i][j] = 1 + f[i - 1][j - 1]
                    else:
                        f[i][j] = max(f[i][j - 1], f[i - 1][j])
            return f[-1][-1]
        return lcs(s, s[::-1])
```

- dfs + cache

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

- dp

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        f = [[0] * n for r in range(n)]
        for i in range(n - 1, -1, -1):
            f[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1] + 2
                else:
                    f[i][j] = max(f[i + 1][j], f[i][j - 1])
        return f[0][n - 1]
```


### 1312. Minimum Insertion Steps to Make a String Palindrome

- dfs + cache

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        @cache
        def dfs(i, j):
            if i >= j:
                return 0
            if s[i] == s[j]:
                return dfs(i + 1, j - 1)
            return min(dfs(i + 1, j), dfs(i, j - 1)) + 1
        return dfs(0, len(s) - 1)
```

- dp

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        f = [[0] * n for r in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1]
                else:
                    f[i][j] = min(f[i + 1][j], f[i][j - 1]) + 1
        return f[0][n - 1]
```

- lcs

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

### 1771. Maximize Palindrome Length From Subsequences

```python
class Solution:
    def longestPalindrome(self, word1: str, word2: str) -> int:
        s = word1 + word2
        n = len(s)
        res = 0
        f = [[0] * n for r in range(n)]
        for i in range(n - 1, -1, -1):
            f[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1] + 2
                    if i < len(word1) <= j:
                        res = max(res, f[i][j])
                else:
                    f[i][j] = max(f[i + 1][j], f[i][j - 1])
        return res
```

### 1682. Longest Palindromic Subsequence II

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        @cache
        def dfs(i, j, prev):
            if i >= j:
                return 0
            ans = 0
            if s[i] == s[j] and s[i] != prev:
                ans = dfs(i + 1, j - 1, s[i]) + 2
            else:
                ans = max(dfs(i + 1, j, prev), dfs(i, j - 1, prev))
            return ans
        return dfs(0, len(s) - 1, None)
```

### 1216. Valid Palindrome III

```python
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        @cache
        def dfs(i, j):
            if i > j:
                return 0
            if i == j:
                return 1
            if s[i] == s[j]:
                return dfs(i + 1, j - 1) + 2
            else:
                return max(dfs(i + 1, j), dfs(i, j - 1))
        return len(s) - dfs(0, len(s) - 1) <= k
```

### 1246. Palindrome Removal

```python
class Solution:
    def minimumMoves(self, arr: List[int]) -> int:
        @cache
        def dfs(i, j):
            if i == j:
                return 1
            if i + 1 == j:
                return 1 if arr[i] == arr[j] else 2
            res = inf
            if arr[i] != arr[j]:
                for k in range(i, j):
                    res = min(res, dfs(i, k) + dfs(k + 1, j))
            else:
                res = dfs(i + 1, j - 1)
                for k in range(i + 1, j):
                    res = min(res, dfs(i, k) + dfs(k + 1, j))
            return res 
        return dfs(0, len(arr) - 1)
```

### 730. Count Different Palindromic Subsequences

```python
class Solution:
    def countPalindromicSubsequences(self, s: str) -> int:
        mod = 10 ** 9 + 7
        @cache
        def dfs(l, r):
            if l >= r:
                return int(l == r)
            res = 0
            for c in 'abcd':
                i, j = s.find(c, l, r + 1), s.rfind(c, l, r + 1)
                if i == -1:
                    continue 
                res += (1 if i == j else 2) + dfs(i + 1, j - 1)
                res %= mod 
            return res 
        return dfs(0, len(s) - 1)
```

## 2 Others

### 5. Longest Palindromic Substring

- range enlarge
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n, res = len(s), ''
        f = [[False] * n for r in range(n)]
        for r in range(n):
            for l in range(r + 1):
                if s[l] == s[r] and (r - l + 1 <= 3 or f[l + 1][r - 1]):
                    f[l][r] = True
                    res = max(res, s[l: r + 1], key = len)
        return res
```

### 3040. Maximum Number of Operations With the Same Score II

```python
class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        @cache
        def dfs(i, j, target):
            if i >= j:
                return 0
            res = 0
            if nums[i] + nums[i + 1] == target:
                res = max(res, dfs(i + 2, j, target) + 1)
            if nums[j - 1] + nums[j] == target:
                res = max(res, dfs(i, j - 2, target) + 1)
            if nums[i] + nums[j] == target:
                res = max(res, dfs(i + 1, j - 1, target) + 1)
            return res

        n = len(nums)
        res1 = dfs(2, n - 1, nums[0] + nums[1]) 
        res2 = dfs(0, n - 3, nums[-2] + nums[-1]) 
        res3 = dfs(1, n - 2, nums[0] + nums[-1]) 
        return max(res1, res2, res3) + 1
```

### 1770. Maximum Score from Performing Multiplication Operations

```python
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        m = len(multipliers)
        @cache
        def dfs(l, r, i):
            if i == m:
                return 0
            res = -inf
            res = max(res, nums[l] * multipliers[i] + dfs(l + 1, r, i + 1))
            res = max(res, nums[r] * multipliers[i] + dfs(l, r - 1, i + 1))
            return res
        return dfs(0, len(nums) - 1, 0)
```

### 1039. Minimum Score Triangulation of Polygon

- dfs + cache

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        @cache
        def dfs(i, j):
            if i + 1 == j:
                return 0
            res = inf 
            for k in range(i + 1, j):
                res = min(res, dfs(i, k) + dfs(k, j) + values[i] * values[k] * values[j])
            return res
        return dfs(0, len(values) - 1)
```

- dp

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        n = len(values)
        f = [[0] * n for r in range(n)]
        for i in range(n - 3, -1, -1):
            for j in range(i + 2, n):
                res = inf 
                for k in range(i + 1, j):
                    res = min(res, f[i][k] + f[k][j] + values[i] * values[j] * values[k])
                f[i][j] = res 
        return f[0][n - 1]
```

### 1130. Minimum Cost Tree From Leaf Values

```python
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        @cache
        def dfs(i, j):
            if i == j:
                return 0
            res = inf
            for k in range(i, j):
                res = min(res, dfs(i, k) + dfs(k + 1, j) + max(arr[i:k+1]) * max(arr[k+1:j+1]))
            return res
        return dfs(0, len(arr) - 1)
```

### 96. Unique Binary Search Trees

```python
class Solution:
    def numTrees(self, n: int) -> int:
        @cache
        def dfs(start, end):
            if start > end:
                return [None]
            res = []
            for i in range(start, end + 1):
                left = dfs(start, i - 1)
                right = dfs(i + 1, end)
                for l in left:
                    for r in right:
                        root = TreeNode(i)
                        root.left = l
                        root.right = r
                        res.append(root)
            return res
        return dfs(1, n)
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


### 2464. Minimum Subarrays in a Valid Split

```python
class Solution:
    def validSubarraySplit(self, nums: List[int]) -> int:
        # [   2,   6,   3,   4,   3]
        # [0, 1,   1,   2,   1,   2]
        # [   1,   2,   1]
        # [0, inf, inf, inf]
        n = len(nums)
        dp = [inf] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                if gcd(nums[i - 1], nums[j - 1]) > 1:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
        return -1 if dp[n] == inf else dp[n]
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

## job schedule dp

### 2008. Maximum Earnings From Taxi

```python
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        f, d = [0] * (n + 1), defaultdict(list) # max profit at i
        for s, e, t in rides:
            d[e].append((s, t))

        for e in range(1, n + 1):
            f[e] = f[e - 1] # profit for not choosing e
            if e in d: # choose e
                for s2, t2 in d[e]:
                    f[e] = max(f[e], e - s2 + t2 + f[s2])      
        return f[-1]
```

### others


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