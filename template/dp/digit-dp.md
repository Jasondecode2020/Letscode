## Always use dfs and template

- 2 types, with leading zerp or without leading zero
- the state is prev state, is_num for prev has set a num, limit for lower and upper boundary

* [357. Count Numbers with Unique Digits](#357-count-numbers-with-unique-digits)
* [233. Number of Digit One](#233-number-of-digit-one)
* [面试题 17.06. Number Of 2s In Range LCCI](#面试题-1706-number-of-2s-in-range-lcci)
* [1067. Digit Count in Range](#1067-digit-count-in-range)
* [2376. Count Special Integers](#2376-count-special-integers)
* [1012. Numbers With Repeated Digits](#1012-numbers-with-repeated-digits)
* [902. Numbers At Most N Given Digit Set](#902-numbers-at-most-n-given-digit-set)
* [2999. Count the Number of Powerful Integers](#2999-count-the-number-of-powerful-integers)
* [2827. Number of Beautiful Integers in the Range](#2827-number-of-beautiful-integers-in-the-range)
* [2801. Count Stepping Numbers in Range](#2801-count-stepping-numbers-in-range)
* [2719. Count of Integers](#2719-count-of-integers)
* [600. Non-negative Integers without Consecutive Ones](#600-non-negative-integers-without-consecutive-ones)
* [788. Rotated Digits](#788-rotated-digits)

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

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        low = str(0)
        high = str(10 ** n - 1)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, mask, limit_low, limit_high, is_num):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, mask, True, False, False)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                if (1 << d) & mask == 0:
                    res += f(i + 1, (1 << d) | mask, limit_low and d == lo, limit_high and d == hi, True)
            return res
        return f(0, 0, True, True, False) + 1
```

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

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        low = str(0)
        high = str(n)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, val):
            if i == n:
                if is_num:
                    return val
                return 0
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, 0)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, val + int(d == 1))
            return res
        res = f(0, True, True, False, 0)
        return res
```


### 面试题 17.06. Number Of 2s In Range LCCI

```python
class Solution:
    def numberOf2sInRange(self, n: int) -> int:
        low = str(0)
        high = str(n)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, val):
            if i == n:
                if is_num:
                    return val
                return 0
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, 0)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, val + int(d == 2))
            return res
        res = f(0, True, True, False, 0)
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

### 2376. Count Special Integers
f
- v1 template 

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
- v2 template

```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        low = str(1)
        high = str(n)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, mask, limit_low, limit_high, is_num):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, mask, True, False, False)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                if (1 << d) & mask == 0:
                    res += f(i + 1, (1 << d) | mask, limit_low and d == lo, limit_high and d == hi, True)
            return res
        return f(0, 0, True, True, False)
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

```python
class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
        num = n 
        low = str(1)
        high = str(n)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, mask, limit_low, limit_high, is_num):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, mask, True, False, False)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                if (1 << d) & mask == 0:
                    res += f(i + 1, (1 << d) | mask, limit_low and d == lo, limit_high and d == hi, True)
            return res
        return num - f(0, 0, True, True, False)
```

### 902. Numbers At Most N Given Digit Set

```python
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        @cache
        def f(i, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = f(i + 1, False, False) if not is_num else 0
            up = s[i] if is_limit else '9'
            for d in digits:
                if d > up:
                    break
                res += f(i + 1, is_limit and d == up, True)
            return res

        s = str(n)
        return f(0, True, False)
```

```python
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        low = str(1)
        high = str(n)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False)
            # range of the ith number, i start from 0
            lo = low[i] if limit_low else '0'
            hi = high[i] if limit_high else '9'
            d0 = 0 if is_num else 1
            for d in digits:
                if d > hi:
                    break
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True)
            return res
        return f(0, True, True, False)
```

### 2999. Count the Number of Powerful Integers

- without considering consider leading zeros

```python
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        low = str(start)
        high = str(finish)
        n = len(high)
        low = '0' * (n - len(low)) + low
        diff = n - len(s)
        @cache
        def f(i, limit_low, limit_high):
            if i == n:
                return 1
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            res = 0
            if i < diff:
                # constraint of limit
                for d in range(lo, min(hi, limit) + 1):
                    res += f(i + 1, limit_low and d == lo, limit_high and d == hi)
            else:
                # must fill int(s[i - diff])
                x = int(s[i - diff])
                if lo <= x <= min(hi, limit):
                    res = f(i + 1, limit_low and x == lo, limit_high and x == hi)
            return res
        return f(0, True, True)
```
- consider leading zeros

```python
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        low = str(start)
        high = str(finish)
        n = len(high)
        low = '0' * (n - len(low)) + low
        diff = n - len(s)
        @cache
        def f(i, limit_low, limit_high, is_num):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            if i < diff:
                # constraint of limit
                for d in range(max(lo, d0), min(hi, limit) + 1):
                    res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True)
            else:
                # must fill int(s[i - diff])
                x = int(s[i - diff])
                if lo <= x <= min(hi, limit):
                    res = f(i + 1, limit_low and x == lo, limit_high and x == hi, True)
            return res
        return f(0, True, True, False)
```

### 2827. Number of Beautiful Integers in the Range

- consider leading zero, v2 template

```python
class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        low = str(low)
        high = str(high)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, val, diff):
            if i == n:
                return int(is_num and val == 0 and diff == 0)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, val, diff)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                # d % 2 * 2 - 1 to check if odd + 1 else -1, val to get the final number
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, (val * 10 + d) % k, diff + d % 2 * 2 - 1) 
            return res
        return f(0, True, True, False, 0, 0)
```

### 2801. Count Stepping Numbers in Range

```python
class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        low = str(low)
        high = str(high)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        mod = 10 ** 9 + 7
        @cache
        def f(i, limit_low, limit_high, is_num, pre):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, pre)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                if not is_num or abs(d - pre) == 1:
                    res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, d)
            return res
        return f(0, True, True, False, 0) % mod
```

### 2719. Count of Integers

```python
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10 ** 9 + 7
        low = str(num1)
        high = str(num2)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, val):
            if i == n:
                if is_num:
                    return int(min_sum <= val <= max_sum)
                return 0
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, 0)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, val + d)
            return res
        res = f(0, True, True, False, 0)
        return res % mod
```

### 600. Non-negative Integers without Consecutive Ones

```python
class Solution:
    def findIntegers(self, n: int) -> int:
        low = str(0)
        high = str(bin(n)[2:])
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, pre):
            if i == n:
                return int(is_num)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, pre)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 1
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                if d == 1 and pre: continue
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, d == 1)
            return res
        res = f(0, True, True, False, False) + 1
        return res
```

### 788. Rotated Digits

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        diff = {2, 5, 6, 9}
        same = {1, 0, 8}
        low = str(1)
        high = str(n)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, has_diff):
            if i == n:
                return int(is_num and has_diff)
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, False)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                if d in diff:
                    res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, True)
                elif d in same:
                    res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, has_diff)
            return res
        return f(0, True, True, False, False)
```

### 2843. Count Symmetric Integers

```python
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        # didn't improve the time, need to optimize
        def check(val):
            s = str(val)
            n = len(s)
            l = list(int(i) for i in s)
            return sum(l[: n // 2]) == sum(l[n // 2:]) and n % 2 == 0
        low = str(low)
        high = str(high)
        n = len(high)
        low = '0' * (n - len(low)) + low # alignment of low and high
        @cache
        def f(i, limit_low, limit_high, is_num, val):
            if i == n:
                print(val)
                return int(is_num and check(val))
            res = 0
            if not is_num and low[i] == '0':
                res += f(i + 1, True, False, False, val)
            # range of the ith number, i start from 0
            lo = int(low[i]) if limit_low else 0
            hi = int(high[i]) if limit_high else 9
            d0 = 0 if is_num else 1
            for d in range(max(lo, d0), hi + 1):
                res += f(i + 1, limit_low and d == lo, limit_high and d == hi, True, val * 10 + d)
            return res
        return f(0, True, True, False, 0)
```

