### 233. Number of Digit One

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        s = str(n)
        N, res = len(s), 0
        for i in range(1, N + 1):
            high = n // pow(10, i)
            low = pow(10, i - 1)
            res += high * low 
            if s[N - i] > '1':
                res += low 
            elif s[N - i] == '1':
                res += n % low + 1
        return res
```

### 1067. Digit Count in Range

```python
class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:
        def check(n):
            s = str(n)
            N, res = len(s), 0
            if d != 0:
                for i in range(1, N + 1):
                    high = n // pow(10, i)
                    low = pow(10, i - 1)
                    res += high * low 
                    if s[N - i] > str(d):
                        res += low 
                    elif s[N - i] == str(d):
                        res += n % low + 1
            else:
                for i in range(1, N):
                    high = n // pow(10, i)
                    low = pow(10, i - 1)
                    res += (high - 1) * low 
                    if s[N - i] > str(d):
                        res += low 
                    elif s[N - i] == str(d):
                        res += n % low + 1
            return res
        return check(high) - check(low - 1)
```

### 357. Count Numbers with Unique Digits

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        def check(n):
            if n == 0:
                return 1
            res = 9
            for i in range(9, 9 - n + 1, -1):
                res *= i
            return res
        return sum(check(i) for i in range(n + 1))
```

### 2376. Count Special Integers

```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        @cache
        def f(i, mask, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(1 - int(is_num), up + 1):
                if (1 << d) & mask == 0:
                    res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
            return res

        s = str(n)
        return f(0, 0, True, False)
```

### 1012. Numbers With Repeated Digits

```python
class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
        @cache
        def f(i, mask, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(1 - int(is_num), up + 1):
                if (1 << d) & mask == 0:
                    res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
            return res

        s = str(n)
        return n - f(0, 0, True, False)
```