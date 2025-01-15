## template

### https://www.youtube.com/watch?v=V5-7GzOfADQ (best video about kmp)

```python
# consider using nxt = [-1] * n 
def nxt(s):
    nxt, j = [-1], -1
    for i in range(len(s)):
        while j >= 0 and s[i] != s[j]:
            j = nxt[j]
        j += 1
        nxt.append(j)
    return nxt
```

- aaa, nxt = [-1, 0, 1, 2]
- abc, nxt = [-1, 0, 0, 0]

## Question list(12)

* [28. Find the Index of the First Occurrence in a String](#28-find-the-index-of-the-first-occurrence-in-a-string)
* [796. Rotate String](#796-rotate-string)
* [1392. Longest Happy Prefix](#1392-longest-happy-prefix)
* [3036. Number of Subarrays That Match a Pattern II](#3036-number-of-subarrays-that-match-a-pattern-ii)
* [3037. Find Pattern in Infinite Stream II](#3037-find-pattern-in-infinite-stream-ii)

* [459. Repeated Substring Pattern](#459-repeated-substring-pattern)
* [1668. Maximum Repeating Substring](#1668-maximum-repeating-substring)
* [686. Repeated String Match](#686-repeated-string-match)
* [214. Shortest Palindrome](#214-shortest-palindrome)

* [1764. Form Array by Concatenating Subarrays of Another Array](#1764-form-array-by-concatenating-subarrays-of-another-array)
* [3008. Find Beautiful Indices in the Given Array II](#3008-find-beautiful-indices-in-the-given-array-ii)(kmp + binary search)
* [1397. Find All Good Strings](#1397-find-all-good-strings) 2666 (TODO: kmp + digit dp)

### 28. Find the Index of the First Occurrence in a String

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 
        
        s, p = haystack, needle
        nxt = nxt(p)
        j = 0
        for i in range(len(s)):
            while j >= 0 and s[i] != p[j]:
                j = nxt[j]
            j += 1
            if j == len(p):
                return i - len(p) + 1
        return -1
```

### 796. Rotate String

```python
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 
        
        s, p = s * 2, goal
        nxt = nxt(p)
        j = 0
        for i in range(len(s)):
            while j >= 0 and s[i] != p[j]:
                j = nxt[j]
            j += 1
            if j == len(p):
                return True 
        return False 
```

### 1392. Longest Happy Prefix

```python
class Solution:
    def longestPrefix(self, s: str) -> str:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 
        k = nxt(s)[-1]
        return s[:k]
```

### 3036. Number of Subarrays That Match a Pattern II

```python
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 
            
        s = [-1 if x > y else 0 if x == y else 1 for x, y in pairwise(nums)]
        j, res = 0, 0
        nxt = nxt(pattern)
        for i in range(len(s)):
            while j >= 0 and s[i] != pattern[j]:
                j = nxt[j]
            j += 1
            if j == len(pattern):
                res += 1
                j = nxt[j]
        return res
```

### 3037. Find Pattern in Infinite Stream II

```python
class Solution:
    def findPattern(self, stream: Optional['InfiniteStream'], pattern: List[int]) -> int:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 

        p = pattern
        nxt = nxt(p)
        j = 0
        i = -1
        while True:
            b = stream.next()
            i += 1
            while j >= 0 and b != p[j]:
                j = nxt[j]
            j += 1
            if j == len(p):
                return i - j + 1
```


### 459. Repeated Substring Pattern

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt
        k = nxt(s)[-1]
        if k == 0:
            return False
        n = len(s)
        k = gcd(k, n)
        return s == s[:k] * (n // k)
```

### 1668. Maximum Repeating Substring

- here use kmp function name, not nxt, otherwise there is error of same name as nxt in while loop
```python
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        s, p = sequence, word
        n = len(s)
        def kmp(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 

        res = 0
        def check(p):
            j = 0
            for i in range(len(s)):
                while j >= 0 and s[i] != p[j]:
                    j = nxt[j]
                j += 1
                if j == m:
                    return True
            return False

        while True:
            nxt = kmp(p)
            m = len(p)
            if check(p):
                res += 1
                p += word 
            else:
                break
        return res 
```

### 686. Repeated String Match

```python
class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        mn_a = a * ceil(len(b) / len(a))
        mx_a = a * (ceil(len(b) / len(a)) + 1)
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt
        nxt = nxt(b)
        def check(s):
            j = 0
            for i in range(len(s)):
                while j >= 0 and s[i] != b[j]:
                    j = nxt[j]
                j += 1
                if j == len(b):
                    return True
            return False
        if check(mn_a):
            return ceil(len(b) / len(a))
        elif check(mx_a):
            return ceil(len(b) / len(a)) + 1
        return -1
```

### 214. Shortest Palindrome

```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt
        nxt = nxt(s + '#' + s[::-1])
        k = nxt[-1]
        return s[::-1] + s[k:]
```

### 1764. Form Array by Concatenating Subarrays of Another Array

- not good

```python
class Solution:
    def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 

        res = []
        for group in groups:
            nxt_arr = nxt(group)
            j = 0
            for i in range(len(nums)):
                while j >= 0 and nums[i] != group[j]:
                    j = nxt_arr[j]
                j += 1
                if j == len(group):
                    if not res or sum(res[-1]) <= i - j + 1:
                        res.append((i - j + 1, len(group)))
                        break
                    j = 0       
        return len(res) == len(groups)
```

- good

```python
class Solution:
    def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
        i = j = 0
        while i < len(groups) and j < len(nums):
            if nums[j: j + len(groups[i])] == groups[i]:
                j += len(groups[i])
                i += 1
            else:
                j += 1
        return i == len(groups)
```

### 3008. Find Beautiful Indices in the Given Array II

```python
class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # index_a = [16,33, 60], index_b = [4, 18]
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 
        
        nxt_a, nxt_b = nxt(a), nxt(b)
        def check(p, nxt):
            j = 0
            res = []
            for i in range(len(s)):
                while j >= 0 and s[i] != p[j]:
                    j = nxt[j]
                j += 1
                if j == len(p):
                    res.append(i - j + 1)
                    j = nxt[j]
            return res 
        arr_a, arr_b = check(a, nxt_a), check(b, nxt_b)
        
        res = []
        for n in arr_a:
            j = bisect_left(arr_b, n - k) # find the smallest possible value then use abs(n - arr_b[j]) <= k
            if j < len(arr_b) and abs(n - arr_b[j]) <= k:
                res.append(n)
        return res
```

### 1397. Find All Good Strings

- learn digit dp first 

```python
class Solution:
    def findGoodStrings(self, n: int, s1: str, s2: str, evil: str) -> int:
        @cache
        def f(i, is_limit, s, hit):
            if hit == len(evil):
                return 0
            if i == len(s):
                return 1
            res = 0
            high = s[i] if is_limit else 'z'
            for d in range(97, ord(high) + 1):
                c = chr(d)
                j = hit
                while j >= 0 and c != evil[j]:
                    j = nxt[j]
                j += 1
                res += f(i + 1, (is_limit and c == high), s, j)
                res %= mod
            return res

        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt

        mod = 10 ** 9 + 7
        nxt = nxt(evil)
        return (f(0, True, s2, 0) - f(0, True, s1, 0) + int(evil not in s1)) % mod
```