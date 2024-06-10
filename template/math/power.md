### 372. Super Pow

```python
class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        m = 1337
        res = 1
        for n in b[::-1]:
            res = res * pow(a, n, m) % m
            a = pow(a, 10, m) % m
        return res
```