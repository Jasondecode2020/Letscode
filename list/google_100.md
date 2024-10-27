## Question list

* [247. Strobogrammatic Number II](#247-strobogrammatic-number-ii)

### 247. Strobogrammatic Number II

```python
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        pairs = ["11","69","88","96"]
        def dfs(x):
            if x == 0:
                return ['']
            if x == 1:
                return ["0","1","8"]
            res = []
            for num in dfs(x - 2):
                for a, b in pairs:
                    res.append(a + num + b)
                if x != n:
                    res.append('0' + num + '0')
            return res 
        return dfs(n)
```

### 248. Strobogrammatic Number III

```python
class Solution:
    def strobogrammaticInRange(self, low: str, high: str) -> int:
        pairs = ["11","69","88","96"] # '00'
        def dfs(x):
            if x == 0:
                return ['']
            if x == 1:
                return ["0","1","8"]
            res = []
            for s in dfs(x - 2):
                for a, b in pairs:
                    res.append(a + s + b)
                if x != n:
                    res.append('0' + s + '0')
            return res 
        n1, n2 = len(low), len(high)
        res = 0
        for n in range(n1, n2 + 1):
            for s in dfs(n):
                if int(low) <= int(s) <= int(high):
                    res += 1
        return res 
```