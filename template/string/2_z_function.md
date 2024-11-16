## template
- link: https://cp-algorithms.com/string/z-function.html
- video: https://www.youtube.com/watch?v=CpZh4eF8QBw&t=162s
- prepare needle + '#' + haystack = s, then find if needle inside haystack

```python
def z_func(s):
    n = len(s)
    z, l, r = [0] * n, 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z

def z_function_trivial(s):
    n = len(s)
    z = [0] * n
    for i in range(1, n):
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
    return z
```

For example, here are the values of the Z-function computed for different strings:

"aaaaa" -  $[0, 4, 3, 2, 1]$ 
"aaabaab" -  $[0, 2, 1, 0, 2, 1, 0]$ 
"abacaba" -  $[0, 0, 1, 0, 3, 0, 1]$ 

## Question list

* [28. Find the Index of the First Occurrence in a String](#28-find-the-index-of-the-first-occurrence-in-a-string)
* [796. Rotate String](#796-rotate-string)
* [2223. Sum of Scores of Built Strings](#2223-sum-of-scores-of-built-strings)
* [214. Shortest Palindrome](#214-shortest-palindrome)
* [1392. Longest Happy Prefix](#1392-longest-happy-prefix)
* [3031. Minimum Time to Revert Word to Initial State II]()

### 28. Find the Index of the First Occurrence in a String

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        def z_func(s):
            n = len(s)
            z, l, r = [0] * n, 0, 0
            for i in range(1, n):
                if i < r:
                    z[i] = min(r - i, z[i - l]) # 'aaaaaaaaa'
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] > r:
                    l, r = i, i + z[i]
            return z

        s = needle + '#' + haystack
        z = z_func(s)
        for i, n in enumerate(z):
            if n == len(needle):
                return i - len(needle) - 1
        return -1
```

### 796. Rotate String

- prepare goal + '#' as pattern
- check pattern as prefix of 2 * s

```python
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        text = goal + '#' + 2 * s
        def z_func(s):
            n = len(s)
            z, l, r = [0] * n, 0, 0
            for i in range(1, n):
                if i < r:
                    z[i] = min(r - i, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] > r:
                    l, r = i, i + z[i]
            return z
        z = z_func(text)
        for i, n in enumerate(z):
            if n == len(goal) and len(goal) == len(s):
                return True
        return False
```

### 2223. Sum of Scores of Built Strings

- directly use

```python
class Solution:
    def sumScores(self, s: str) -> int:
        def z_func(s):
            n = len(s)
            z, l, r = [0] * n, 0, 0
            for i in range(1, n):
                if i < r:
                    z[i] = min(r - i, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] > r:
                    l, r = i, i + z[i]
            return z
        z = z_func(s)
        res = sum(z)
        return res + len(s)
```

### 214. Shortest Palindrome

```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        text = s + '#' + s[::-1]
        def z_func(s):
            n = len(s)
            z, l, r = [0] * n, 0, 0
            for i in range(1, n):
                if i < r:
                    z[i] = min(r - i, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] > r:
                    l, r = i, i + z[i]
            return z

        z = z_func(text)
        idx = 0
        nums = z[len(s) + 1: ] # 取'#'之后的翻转字符串的z数组
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == len(nums) - i:
                idx = i # 倒序找出最长回文串起点
        return text[len(s) + 1: ][:idx] + s # 剩余部分加s
```

### 1392. Longest Happy Prefix

```python
class Solution:
    def longestPrefix(self, s: str) -> str:
        def z_func(s):
            n = len(s)
            z, l, r = [0] * n, 0, 0
            for i in range(1, n):
                if i < r:
                    z[i] = min(r - i, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] > r:
                    l, r = i, i + z[i]
            return z

        z = z_func(s)
        idx = len(z)
        for i in range(len(z) - 1, -1, -1):
            if z[i] == len(z) - i:
                idx = i # 倒序找出最长回文串起点
        return s[idx: ]
```

### 3031. Minimum Time to Revert Word to Initial State II

```python
class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        def z_func(s):
            n = len(s)
            z, l, r = [0] * n, 0, 0
            for i in range(1, n):
                if i < r:
                    z[i] = min(r - i, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] > r:
                    l, r = i, i + z[i]
            return z 
        n = len(word)
        res = z_func(word)
        for i in range(k, n, k):
            if res[i] == n - i:
                return i // k
        return n // k if n % k == 0 else n // k + 1
```