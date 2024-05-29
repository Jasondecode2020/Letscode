### 816. Ambiguous Coordinates

```python
class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        def check(s):
            res = []
            if s[0] != '0' or s == '0':
                res.append(s)
            for i in range(1, len(s)):
                if i > 1 and s[0] == '0' or s[-1] == '0':
                    continue
                res.append(s[:i] + '.' + s[i:])
            return res

        s = s[1:-1]
        n = len(s)
        res = []
        for i in range(1, n):
            left, right = check(s[:i]), check(s[i:])
            if not left or not right:
                continue 
            for a, b in product(left, right):
                res.append('(' + a + ', ' + b + ')')
        return res
```