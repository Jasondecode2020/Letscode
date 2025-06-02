## template

```python
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
```

### question list: (16)

* [1143. Longest Common Subsequence](#1143-longest-common-subsequence)
* [583. Delete Operation for Two Strings](#583-delete-operation-for-two-strings)
* [712. Minimum ASCII Delete Sum for Two Strings](#712-minimum-ascii-delete-sum-for-two-strings)
* [72. Edit Distance](#72-edit-distance)
* [97. Interleaving String](#97-interleaving-string)

* [115. Distinct Subsequences](#115-distinct-subsequences)
* [1035. Uncrossed Lines](#1035-uncrossed-lines)
* [1458. Max Dot Product of Two Subsequences](#1458-max-dot-product-of-two-subsequences)
* [718. Maximum Length of Repeated Subarray](#718-maximum-length-of-repeated-subarray)
* [1092. Shortest Common Supersequence](#1092-shortest-common-supersequence)

* [1639. Number of Ways to Form a Target String Given a Dictionary](#1639-number-of-ways-to-form-a-target-string-given-a-dictionary)
* [161. One Edit Distance](#161-one-edit-distance)
* [516. Longest Palindromic Subsequence](#516-longest-palindromic-subsequence)
* [1312. Minimum Insertion Steps to Make a String Palindrome](#1312-minimum-insertion-steps-to-make-a-string-palindrome)
* [44. Wildcard Matching](#44-wildcard-matching)

* [10. Regular Expression Matching](#10-regular-expression-matching)
* [3290. Maximum Multiplication Score](#3290-maximum-multiplication-score)

### 1143. Longest Common Subsequence

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
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
        return lcs(text1, text2)
```

### 583. Delete Operation for Two Strings

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
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
        return len(word1) + len(word2) - 2 * lcs(word1, word2)
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

### 1035. Uncrossed Lines

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
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
        return lcs(nums1, nums2)
```

### 1458. Max Dot Product of Two Subsequences

```python
class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        R, C = len(nums1) + 1, len(nums2) + 1
        f = [[-inf] * C for r in range(R)]
        for r in range(1, R):
            for c in range(1, C):
                f[r][c] = max(f[r - 1][c], f[r][c - 1], f[r - 1][c - 1] + nums1[r - 1] * nums2[c - 1], nums1[r - 1] * nums2[c - 1])
        return f[-1][-1]
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

### 1092. Shortest Common Supersequence 

```python
class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        R, C = len(str1) + 1, len(str2) + 1
        f = [[0] * C for r in range(R)]
        for r in range(1, R):
            f[r][0] += f[r - 1][0] + 1
        for c in range(1, C):
            f[0][c] += f[0][c - 1] + 1
        for r in range(1, R):
            for c in range(1, C):
                if str1[r - 1] != str2[c - 1]:
                    f[r][c] = min(f[r - 1][c] + 1, f[r][c - 1] + 1)
                else:
                    f[r][c] = f[r - 1][c - 1] + 1
        
        res = []
        r, c = R - 1, C - 1
        while r >= 1 and c >= 1:
            if str1[r - 1] == str2[c - 1]:
                res.append(str1[r - 1])
                r -= 1
                c -= 1
            elif f[r][c] == f[r - 1][c] + 1:
                res.append(str1[r - 1])
                r -= 1
            else:
                res.append(str2[c - 1])
                c -= 1
        return str1[: r] + str2[: c] + ''.join(reversed(res))
```

### 1639. Number of Ways to Form a Target String Given a Dictionary

```python
n = len(words[0])
        m = len(target)
        mod = 10 ** 9 + 7
        dp = [[0] * 26 for i in range(n)]
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                dp[j][ord(w[j]) - ord('a')] += 1
        
        @cache
        def dfs(i, j):
            if j == m:
                return 1
            if i == n:
                return 0
            return dfs(i + 1, j + 1) * dp[i][ord(target[j]) - ord('a')] + dfs(i + 1, j)
        return dfs(0, 0) % mod
```

### 161. One Edit Distance

Given two strings s and t, return true if they are both one edit distance apart, otherwise return false.

A string s is said to be one distance apart from a string t if you can:

Insert exactly one character into s to get t.
Delete exactly one character from s to get t.
Replace exactly one character of s with a different character to get t.
 
Example 1:

Input: s = "ab", t = "acb"
Output: true
Explanation: We can insert 'c' into s to get t.

Example 2:

Input: s = "", t = ""
Output: false
Explanation: We cannot get t from s by only one step.
 
Constraints:

0 <= s.length, t.length <= 104
s and t consist of lowercase letters, uppercase letters, and digits.

### 161. One Edit Distance

```python
class Solution:
    def isOneEditDistance(self, s: str, t: str) -> bool:
        ns, nt = len(s), len(t)
        if ns > nt:
            return self.isOneEditDistance(t, s)
        if nt - ns > 1:
            return False

        for i in range(ns):
            if s[i] != t[i]:
                if ns == nt:
                    return s[i + 1:] == t[i + 1:]
                else:
                    return s[i:] == t[i + 1:]
        return ns + 1 == nt
```

### 516. Longest Palindromic Subsequence

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
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
        return LCS(s, s[::-1])
```

### 1312. Minimum Insertion Steps to Make a String Palindrome

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

### 3290. Maximum Multiplication Score

```python
class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        @cache
        def f(i, j):
            if i == n:
                return 0 if j == 4 else -inf 
            res = f(i + 1, j)
            if j < 4:
                res = max(res, f(i + 1, j + 1) + a[j] * b[i])
            return res 
        n = len(b)
        return f(0, 0)
```