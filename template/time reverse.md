### 2514. Count Anagrams

```python
m = 10 ** 9 + 7
@lru_cache(None)
def modInverse(i):
    return pow(i, m - 2, m)

@lru_cache(None)
def Factorial(i):
    if i == 0:
        return 1
    return (i * Factorial(i - 1)) % m

class Solution:
    def countAnagrams(self, s: str) -> int:
        def check(c):
            res = 1
            for k, v in c.items():
                res *= modInverse(Factorial(v))
                res %= m
            return res

        res = 1
        for w in s.split(' '):
            n = len(w)
            c = Counter(w)
            res *= (Factorial(n) * check(c)) % m
            res %= m
        return res
```