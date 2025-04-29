## Always use dfs and template

* [2719. Count of Integers](#2719-count-of-integers)
* [1399. Count Largest Group](#1399-count-largest-group)
* [788. Rotated Digits](#788-rotated-digits)
* [1742. Maximum Number of Balls in a Box](#1742-maximum-number-of-balls-in-a-box)
* [902. Numbers At Most N Given Digit Set](#902-numbers-at-most-n-given-digit-set)

* [600. Non-negative Integers without Consecutive Ones](#600-non-negative-integers-without-consecutive-ones)
* [2376. Count Special Integers](#2376-count-special-integers)
* [357. Count Numbers with Unique Digits](#357-count-numbers-with-unique-digits)
* [1012. Numbers With Repeated Digits](#1012-numbers-with-repeated-digits)
* [3519. Count Numbers with Non-Decreasing Digits](#3519-count-numbers-with-non-decreasing-digits)

* [2827. Number of Beautiful Integers in the Range](#2827-number-of-beautiful-integers-in-the-range)
* [2999. Count the Number of Powerful Integers](#2999-count-the-number-of-powerful-integers)
* [2801. Count Stepping Numbers in Range](#2801-count-stepping-numbers-in-range)
* [3490. Count Beautiful Numbers](#3490-count-beautiful-numbers)
* [233. Number of Digit One](#233-number-of-digit-one)

* [1215. Stepping Numbers](#1215-stepping-numbers)
* [3032. Count Numbers With Unique Digits II](#3032-count-numbers-with-unique-digits-ii)
* [1067. Digit Count in Range](#1067-digit-count-in-range)
* [面试题 17.06. Number Of 2s In Range LCCI](#面试题-1706-number-of-2s-in-range-lcci)
* [1397. Find All Good Strings](#1397)

### 2376. Count Special Integers

```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)
        @cache
        def f(i, mask, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, mask, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9 
            for d in range(low, high + 1):
                if (mask >> d) & 1 == 0:
                    res += f(i + 1, mask | (1 << d), is_limit and d == high, True)
            return res 
        return f(0, 0, True, False)
```

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

### 233. Number of Digit One

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        s = str(n)
        @cache
        def f(i, cnt, is_limit, is_num):
            if i == len(s):
                return cnt 
            res = 0
            if not is_num: # not choose digit
                res = f(i + 1, cnt, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1): # choose digit
                res += f(i + 1, cnt + (d == 1), is_limit and d == high, True)
            return res 
        return f(0, 0, True, False)
```

### 2719. Count of Integers

```python
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        s1, s2 = str(int(num1) - 1), num2
        mod = 10 ** 9 + 7
        @cache
        def f(i, total, is_limit, is_num, s):
            if i == len(s):
                if is_num:
                    return min_sum <= total <= max_sum
                return 0
            res = 0
            if not is_num:
                res = f(i + 1, total, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                res += f(i + 1, total + d, is_limit and d == high, True, s)
            return res 
        return (f(0, 0, True, False, s2) - f(0, 0, True, False, s1)) % mod
        
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
        s = str(10 ** n - 1)
        @cache
        def f(i, mask, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, mask, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9 
            for d in range(low, high + 1):
                if (mask >> d) & 1 == 0:
                    res += f(i + 1, mask | (1 << d), is_limit and d == high, True)
            return res 
        return f(0, 0, True, False) + 1
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
        s = str(n)
        @cache
        def f(i, cnt, is_limit, is_num):
            if i == len(s):
                return cnt 
            res = 0
            if not is_num: # not choose digit
                res = f(i + 1, cnt, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1): # choose digit
                res += f(i + 1, cnt + (d == 2), is_limit and d == high, True)
            return res 
        return f(0, 0, True, False)
```

### 1397. Find All Good Strings

```python
class Solution:
    def findGoodStrings(self, n: int, s1: str, s2: str, evil: str) -> int:
        # digit dp
        @cache
        def f(i, is_limit, s, hit):
            if hit == len(evil):
                return 0
            if i == len(s):
                return 1
            res = 0
            high = s[i] if is_limit else 'z'
            for d in range(97, ord(high) + 1):
                c = chr(d)
                j = hit 
                while j >= 0 and c != evil[j]:
                    j = nxt[j]
                j += 1
                if c <= high:
                    res += f(i + 1, is_limit and c == high, s, j)
            return res
        # kmp 
        def nxt(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt 
        
        mod = 10 ** 9 + 7
        nxt = nxt(evil)
        return (f(0, True, s2, 0) - f(0, True, s1, 0) + int(evil not in s1)) % mod
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

### 1012. Numbers With Repeated Digits

```python
class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
        s = str(n)
        @cache
        def f(i, mask, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, mask, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9 
            for d in range(low, high + 1):
                if (mask >> d) & 1 == 0:
                    res += f(i + 1, mask | (1 << d), is_limit and d == high, True)
            return res 
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

### 3519. Count Numbers with Non-Decreasing Digits

```python
class Solution:
    def countNumbers(self, l: str, r: str, b: int) -> int:
        def check(n):
            res = ''
            while n:
                n, m = divmod(n, b)
                res = str(m) + res 
            return res

        s1, s2 = check(int(l) - 1), check(int(r))
        mod = 10 ** 9 + 7
        @cache
        def f(i, pre, is_limit, is_num, s):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, pre, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else b - 1
            for d in range(low, high + 1):
                if d >= pre:
                    res += f(i + 1, d, is_limit and d == high, True, s)
            return res 
        return (f(0, -1, True, False, s2) - f(0, -1, True, False, s1)) % mod
```

### 902. Numbers At Most N Given Digit Set

```python
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        s = str(n)
        @cache
        def f(i, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                if str(d) in digits:
                    res += f(i + 1, is_limit and d == high, True)
            return res 
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

### 3032. Count Numbers With Unique Digits II

```python
class Solution:
    def numberCount(self, a: int, b: int) -> int:
        s1, s2 = str(a - 1), str(b)
        @cache
        def f(i, mask, is_limit, is_num, s):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, mask, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9 
            for d in range(low, high + 1):
                if (mask >> d) & 1 == 0:
                    res += f(i + 1, mask | (1 << d), is_limit and d == high, True, s)
            return res 
        return f(0, 0, True, False, s2) - f(0, 0, True, False, s1)
```

### 1067. Digit Count in Range

```python
class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:
        s1, s2 = str(low - 1), str(high)
        @cache
        def f(i, cnt, is_limit, is_num, s):
            if i == len(s):
                return cnt 
            res = 0
            if not is_num:
                res = f(i + 1, cnt, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9 
            for digit in range(low, high + 1):
                res += f(i + 1, cnt + (digit == d), is_limit and digit == high, True, s)
            return res 
        return f(0, 0, True, False, s2) - f(0, 0, True, False, s1)
```

### 2999. Count the Number of Powerful Integers

- without considering consider leading zeros

```python
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        t = len(s)
        x = int(s)
        def g(num):
            if num < x:
                return 0
            num = str(num)
            m = len(num)
            flag = int(num[-t:]) < x
            @cache
            def f(i, is_limit, is_num):
                if i == m - t:
                    if is_limit and flag:
                        return 0
                    return 1
                res = 0
                if not is_num:
                    res = f(i+1, False, False)
                low = 0 if is_num else 1
                high = int(num[i]) if is_limit else 9
                for d in range(low, min(limit, high) + 1):
                    res += f(i + 1, is_limit and d == high, True)
                return res
            return f(0, True, False)
        return g(finish) - g(start - 1)

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

### 3490. Count Beautiful Numbers

```python
class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        s1, s2 = str(l - 1), str(r)
        @cache
        def f(i, prod_, sum_, is_limit, is_num, s):
            if i == len(s):
                return int(is_num and prod_ % sum_ == 0)
            res = 0
            if not is_num:
                res = f(i + 1, prod_, sum_, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                res += f(i + 1, prod_ * d, sum_ + d, is_limit and d == high, True, s)
            return res 
        return f(0, 1, 0, True, False, s2) - f(0, 1, 0, True, False, s1)
```

### 1399. Count Largest Group

```python
class Solution:
    def countLargestGroup(self, n: int) -> int:
        d = Counter()
        for i in range(1, n + 1):
            res = sum(int(n) for n in str(i))
            d[res] += 1
        mx = max(d.values())
        return list(d.values()).count(mx)

class Solution:
    def countLargestGroup(self, n: int) -> int:
        s = str(n)
        @cache
        def f(i, total, is_limit, is_num):
            if i == len(s):
                return 1 if total == 0 else 0
            res = 0
            if not is_num:
                res = f(i + 1, total, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                res += f(i + 1, total - d, is_limit and d == high, True)
            return res 

        max_cnt = res = 0
        for target in range(1, 50):
            cnt = f(0, target, True, False)
            if cnt > max_cnt:
                max_cnt = cnt 
                res = 1
            elif cnt == max_cnt:
                res += 1
        return res 
```

### 2827. Number of Beautiful Integers in the Range

- consider leading zero, v2 template

```python
class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        s1, s2 = str(low - 1), str(high)
        @cache
        def f(i, val, diff, is_limit, is_num, s):
            if i == len(s):
                return int(is_num and diff == 0 and val == 0)
            res = 0
            if not is_num:
                res = f(i + 1, val, diff, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9 
            for d in range(low, high + 1):
                new_diff = diff + d % 2 * 2 - 1
                new_val = (val * 10 + d) % k
                res += f(i + 1, new_val, new_diff, is_limit and d == high, True, s)
            return res 
        return f(0, 0, 0, True, False, s2) - f(0, 0, 0, True, False, s1)

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
        s1, s2 = str(int(low) - 1), high 
        mod = 10 ** 9 + 7
        @cache
        def f(i, pre, is_limit, is_num, s):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, pre, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                if not is_num or abs(d - pre) == 1:
                    res += f(i + 1, d, is_limit and d == high, True, s)
            return res 
        return (f(0, -1, True, False, s2) - f(0, -1, True, False, s1)) % mod

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

### 600. Non-negative Integers without Consecutive Ones

```python
class Solution:
    def findIntegers(self, n: int) -> int:
        s = bin(n)[2:]
        @cache
        def f(i, pre, is_limit, is_num):
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res = f(i + 1, pre, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 1 
            for d in range(low, high + 1):
                if pre and d == 1:
                    continue
                res += f(i + 1, d == 1, is_limit and d == high, True)
            return res 
        return f(0, False, True, False) + 1
```

### 788. Rotated Digits

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        diff = {2, 5, 6, 9}
        valid = {2, 5, 6, 9, 1, 0, 8}
        s = str(n)
        @cache
        def f(i, is_in_diff, is_limit, is_num):
            if i == len(s):
                return is_in_diff
            res = 0
            if not is_num:
                res = f(i + 1, is_in_diff, False, False)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                if d in valid:
                    res += f(i + 1, is_in_diff or d in diff, is_limit and d == high, True)
            return res 
        return f(0, False, True, False)
```

### 1742. Maximum Number of Balls in a Box

```python
class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        @cache
        def f(i, total, is_limit, is_num, s):
            if i == len(s):
                return 1 if total == 0 else 0
            res = 0
            if not is_num:
                res = f(i + 1, total, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                res += f(i + 1, total - d, is_limit and d == high, True, s)
            return res 

        max_cnt = 0
        s1, s2 = str(lowLimit - 1), str(highLimit)
        for target in range(1, 50):
            cnt = f(0, target, True, False, s2) - f(0, target, True, False, s1)
            if cnt > max_cnt:
                max_cnt = cnt 
        return max_cnt 
```

### 1215. Stepping Numbers

```python
class Solution:
    def countSteppingNumbers(self, low: int, high: int) -> List[int]:
        s1, s2 = str(max(int(low) - 1, 0)), str(high)
        @cache
        def f(i, pre, is_limit, is_num, s):
            if i == len(s):
                return [int(pre)] if is_num else []
            res = []
            if not is_num:
                res = f(i + 1, pre, False, False, s)
            low = 0 if is_num else 1
            high = int(s[i]) if is_limit else 9
            for d in range(low, high + 1):
                if not is_num or abs(d - int(pre[-1])) == 1:
                    res.extend(f(i + 1, pre + str(d), is_limit and d == high, True, s))
            return res 
        res2, res1 = f(0, '', True, False, s2), f(0, '', True, False, s1)
        if low == 0:
            res2 += [0]
        res1.sort()
        res2.sort()
        if not res1:
            return res2
        
        x = res1[-1]
        for i, v in enumerate(res2):
            if v == x:
                return res2[i + 1:]
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

