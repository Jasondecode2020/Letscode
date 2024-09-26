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