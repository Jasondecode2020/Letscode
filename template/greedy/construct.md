### 1702. Maximum Binary String After Change

```python
class Solution:
    def maximumBinaryString(self, binary: str) -> str:
        s = binary
        i = s.find('0')
        if i == -1:
            return s
        zero = s.count('0')
        n = len(s)
        return '1' * i + '1' * (zero  - 1) + '0' + '1' * (n - zero - i)
```

### 2311. Longest Binary Subsequence Less Than or Equal to K

```python
class Solution:
    def longestSubsequence(self, s: str, k: int) -> int:
        s = ''.join(reversed(list(s)))
        total = 0
        for i, c in enumerate(s):
            total += int(c) * 2 ** i 
            if total > k:
                return i + s[i:].count('0')
        return len(s)
```

### 906. Super Palindromes

```python
palindrome = [1, 2, 3, 4, 5, 6, 7, 8, 9] # find all panlindromes
for i in range(1, 10000):
    s1 = str(i) + str(i)[::-1]
    palindrome.append(int(s1))
    for mid in range(10):
        s2 = str(i) + str(mid) + str(i)[::-1]
        palindrome.append(int(s2))

class Solution:
    def superpalindromesInRange(self, left: str, right: str) -> int:
        res = []
        for p in palindrome:
            n = p ** 2
            if int(left) <= n <= int(right):
                x = str(n)
                if x == x[::-1]:
                    res.append(x)
        return len(res)
```