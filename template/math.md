### 50. Pow(x, n)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def dfs(x, n):
            if n == 1:
                return x
            res = dfs(x, n // 2)
            if n % 2 == 0:
                return res * res 
            return x * res * res 
        if n == 0:
            return 1
        if n > 0:
            return dfs(x, n)
        else:
            return 1 / dfs(x, abs(n))
```

### 2961. Double Modular Exponentiation

```python
class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        res = []
        for i, (a, b, c, m) in enumerate(variables):
            A = pow(a, b, 10)
            if pow(A, c, m) == target:
                res.append(i)
        return res
```