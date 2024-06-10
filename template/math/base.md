### 504. Base 7

```python
class Solution:
    def convertToBase7(self, num: int) -> str:
        def check(n):
            res = ''
            while n:
                d, m = divmod(n, 7)
                res = str(m) + res 
                n = d
            return res

        if num > 0:
            return check(num)
        elif num < 0:
            return '-' + check(-num)
        else:
            return '0'
```

### 405. Convert a Number to Hexadecimal

```python
class Solution:
    def toHex(self, num: int) -> str:
        s = '0123456789abcdef'
        def check(n):
            res = ''
            while n:
                d, m = divmod(n, 16)
                res = s[m] + res
                n = d
            return res

        if num > 0:
            return check(num)
        elif num < 0:
            return check(2 ** 32 + num)
        else:
            return '0'
```