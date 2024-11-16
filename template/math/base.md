## Question list

* [504. Base 7](#504-base-7)
* [405. Convert a Number to Hexadecimal](#405-convert-a-number-to-hexadecimal)
* [166. Fraction to Recurring Decimal](#166-fraction-to-recurring-decimal)

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

### 166. Fraction to Recurring Decimal

```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        a, b = numerator, denominator
        if a % b == 0:  
            return str(a // b)
        
        res = ''
        if a * b < 0: 
            res += '-' 
        
        a, b = abs(a), abs(b)
        res += str(a // b) + '.'
        a %= b
        d = {}
        while a != 0:
            if a in d:
                u = d[a]  # 如果出现了就是循环小数
                return res[:u] + f'({res[u:]})'
            d[a] = len(res)
            a *= 10
            res += str(a // b)
            a %= b
        return res
```