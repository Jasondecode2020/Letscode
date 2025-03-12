### 761. Special Binary String

```python 
class Solution:
    def makeLargestSpecial(self, s: str) -> str:
        cnt, l = 0, 0
        res = []
        for r in range(len(s)):
            cnt += 1 if s[r] == '1' else -1
            if cnt == 0:
                res.append('1' + self.makeLargestSpecial(s[l + 1: r]) + '0')
                l = r + 1
        return ''.join(sorted(res, reverse = True))
```