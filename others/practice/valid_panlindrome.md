### 125. Valid Palindrome

```python 
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = [c.lower() for c in s if c.isalnum()]
        return s == s[::-1]
```

### 680. Valid Palindrome II

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        i,j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return s[i+1:j+1] == s[i+1:j+1][::-1] or s[i:j] == s[i:j][::-1] 
            i += 1
            j -= 1
        return True
```

### 1216. Valid Palindrome III

```python 
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        @cache
        def f(i, j):
            if i > j:
                return 0
            if i == j:
                return 1 
            if s[i] == s[j]:
                return f(i + 1, j - 1) + 2 
            return max(f(i + 1, j), f(i, j - 1))
        return len(s) - f(0, len(s) - 1) <= k
```

### 2330. Valid Palindrome IV

```python 
class Solution:
    def makePalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        res = 0
        while l < r:
            if s[l] != s[r]:
                res += 1
            l += 1
            r -= 1
        return res <= 2
```