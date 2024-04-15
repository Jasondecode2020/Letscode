### 10. Regular Expression Matching

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        R, C = len(s) + 1, len(p) + 1
        dp = [[False] * C for r in range(R)]
        dp[0][0] = True
        for c in range(2, C):
            dp[0][c] = dp[0][c - 2] and p[c - 1] == '*'
        for r in range(1, R):
            for c in range(1, C):
                if p[c - 1] == '.' or p[c - 1] == s[r - 1]:
                    dp[r][c] = dp[r - 1][c - 1]
                elif p[c - 1] == '*':
                    zero = dp[r][c - 2]
                    many = dp[r - 1][c] and (p[c - 2] == '.' or p[c - 2] == s[r - 1])
                    dp[r][c] = zero or many 
        return dp[-1][-1]
```

### 44. Wildcard Matching

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        R, C = len(s) + 1, len(p) + 1
        dp = [[False] * C for r in range(R)]
        dp[0][0] = True
        for c in range(1, C):
            if p[c - 1] != '*': 
                break
            dp[0][c] = True
        for r in range(1, R):
            for c in range(1, C):
                if p[c - 1] == '?' or p[c - 1] == s[r - 1]:
                    dp[r][c] = dp[r - 1][c - 1]
                elif p[c - 1] == '*':
                    dp[r][c] = dp[r][c - 1] or dp[r - 1][c]
        return dp[-1][-1]
```

### 72. Edit Distance

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        R, C = len(word1) + 1, len(word2) + 1
        dp = [[0] * C for r in range(R)]
        for c in range(1, C):
            dp[0][c] = dp[0][c - 1] + 1
        for r in range(1, R):
            dp[r][0] = dp[r - 1][0] + 1
        for r in range(1, R):
            for c in range(1, C):
                if word1[r - 1] == word2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1]
                else:
                    dp[r][c] = min(dp[r][c - 1], dp[r - 1][c], dp[r - 1][c - 1]) + 1
        return dp[-1][-1]
```

### 97. Interleaving String

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        R, C = len(s1) + 1, len(s2) + 1
        if R + C - 2 != len(s3):
            return False
        dp = [[False] * C for r in range(R)]
        dp[0][0] = True
        for r in range(1, R):
            dp[r][0] = dp[r - 1][0] and (s1[r - 1] == s3[r - 1])
        for c in range(1, C):
            dp[0][c] = dp[0][c - 1] and (s2[c - 1] == s3[c - 1])
        for r in range(1, R):
            for c in range(1, C):
                dp[r][c] = (dp[r][c - 1] and s3[r + c - 1] == s2[c - 1]) or (dp[r - 1][c] and s3[r + c - 1] == s1[r - 1])
        return dp[-1][-1]
```

### 115. Distinct Subsequences

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        R, C = len(s) + 1, len(t) + 1
        dp = [[0] * C for r in range(R)]
        for r in range(R):
            dp[r][0] = 1
        for c in range(1, C):
            for r in range(c, R):
                if s[r - 1] == t[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1] + dp[r - 1][c]
                else:
                    dp[r][c] = dp[r - 1][c]
        return dp[-1][-1]
```

### 583. Delete Operation for Two Strings

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        def LCS(s1, s2):
            R, C = len(s1) + 1, len(s2) + 1
            dp = [[0] * C for i in range(R)]
            for i in range(1, R):
                for j in range(1, C):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = 1 + dp[i - 1][j - 1]
                    else:
                        dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
            return dp[-1][-1]
        lcs = LCS(word1, word2)
        return len(word1) + len(word2) - 2 * lcs
```

### 712. Minimum ASCII Delete Sum for Two Strings

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        R, C = len(s1) + 1, len(s2) + 1
        dp = [[0] * C for r in range(R)]
        for c in range(1, C):
            dp[0][c] += ord(s2[c - 1]) + dp[0][c - 1]
        for r in range(1, R):
            dp[r][0] += ord(s1[r - 1]) + dp[r - 1][0]
            
        for r in range(1, R):
            for c in range(1, C):
                if s1[r - 1] == s2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1]
                else:
                    dp[r][c] = min(dp[r - 1][c] + ord(s1[r - 1]), dp[r][c - 1] + ord(s2[c - 1]))
        return dp[-1][-1]
```

### 718. Maximum Length of Repeated Subarray

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        R, C = len(nums1) + 1, len(nums2) + 1
        dp = [[0] * C for r in range(R)]
        res = 0
        for r in range(1, R):
            for c in range(1, C):
                if nums1[r - 1] == nums2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1] + 1
                    res = max(res, dp[r][c])
        return res
```

### 727. Minimum Window Subsequence

```python
class Solution:
    def minWindow(self, s1: str, s2: str) -> str:
        R, C = len(s1) + 1, len(s2) + 1
        dp = [[0] * C for r in range(R)]
        for c in range(1, C):
            dp[0][c] = inf 
        for r in range(1, R):
            for c in range(1, C):
                if s1[r - 1] == s2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1] + 1
                else:
                    dp[r][c] = dp[r - 1][c] + 1
        min_len = inf 
        end_pos = -1
        for i in range(1, R):
            if dp[i][C - 1] < min_len:
                min_len = dp[i][C - 1]
                end_pos = i
        return '' if min_len == inf else s1[end_pos - min_len: end_pos]
```

### 1035. Uncrossed Lines

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        R, C = len(nums1) + 1, len(nums2) + 1
        dp = [[0] * C for r in range(R)]
        for r in range(1, R):
            for c in range(1, C):
                if nums1[r - 1] != nums2[c - 1]:
                    dp[r][c] = max(dp[r - 1][c], dp[r][c - 1])
                else:
                    dp[r][c] = dp[r - 1][c - 1] + 1
        return dp[-1][-1]
```