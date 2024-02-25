### 1780. Check if Number is a Sum of Powers of Three

```python
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        s = set()
        while n:
            sign = False
            for i in range(15, -1, -1):
                if n - 3 ** i >= 0 and i not in s:
                    n -= 3 ** i 
                    s.add(i)
                    sign = True
                    break
            if not sign:
                break
        return n == 0
```