## combinations

* [357. Count Numbers with Unique Digits](#357-Count-Numbers-with-Unique-Digits)

### 357. Count Numbers with Unique Digits

> if n = 0, return 1
> n = 1, can choose from 1 to 9, return 9 for 1 digit
> n = 2, for 2 digit, first digit can not choose 0, 9 cases, second digit can not choose
> first one, 9 cases, after than will be, 8, 7, 6, ...

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

## math combinations

- use plate to find 

### 2928. Distribute Candies Among Children I
### 2929. Distribute Candies Among Children II
### 2927. Distribute Candies Among Children III

```python
def C2(n):
    return n * (n - 1) // 2 if n >= 1 else 0
class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        return C2(n + 2) - 3 * C2(n + 2 - (limit + 1)) + 3 * C2(n + 2 - 2 * (limit + 1)) - C2(n + 2 - 3 * (limit + 1))
```

### 1573. Number of Ways to Split a String

```python
class Solution:
    def numWays(self, s: str) -> int:
        n = len(s)
        mod = 10 ** 9 + 7
        ones = s.count('1')
        if ones % 3:
            return 0
        k = ones // 3
        if k == 0:
            return ((n - 1) * (n - 2) // 2) % mod
        i = 0
        res = 1
        count = 0
        s = s.strip('0')
        n = len(s)
        while i < n:
            if s[i] == '1':
                count += 1
            if count == k:
                count = 0
                j = i + 1
                while j < n and s[j] != '1':
                    j += 1
                res *= (j - i)
                i = j
            else:
                i += 1
        return res % mod
```
