### 2575. Find the Divisibility Array of a String

```python
class Solution:
    def divisibilityArray(self, word: str, m: int) -> List[int]:
        remainder = 0
        n = len(word)
        res = [0] * n
        for i, c in enumerate(word):
            ans = remainder * 10 + int(c)
            if ans % m == 0:
                res[i] = 1
            remainder = ans % m
        return res
```