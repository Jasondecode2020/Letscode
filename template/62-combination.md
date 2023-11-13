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