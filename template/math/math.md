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

### 829. Consecutive Numbers Sum

```python
class Solution:
    def consecutiveNumbersSum(self, n: int) -> int:
        res, n = 0, 2 * n 
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0 and (n // i - (i - 1)) % 2 == 0:
                res += 1
        return res
```

### 1785. Minimum Elements to Add to Form a Given Sum

```python
class Solution:
    def minElements(self, nums: List[int], limit: int, goal: int) -> int:
        res = goal - sum(nums)
        res = abs(res)
        return ceil(res / limit)
```

### 1093. Statistics from a Large Sample

```python
class Solution:
    def sampleStats(self, count: List[int]) -> List[float]:
        mn, mx, mean, median, mode = inf, -inf, 0, 0, 0
        mx_f = max(count)
        total_num, total_val = 0, 0
        for i, f in enumerate(count):
            if f:
                mn = min(mn, i)
                mx = max(mx, i)
                total_num += f
                total_val += f * i 
            if f == mx_f:
                mode = i
        cnt = 0
        a, b = -1, -1
        for i, f in enumerate(count):
            if f:
                cnt += f
                if total_num % 2 == 0:
                    if a == -1 and cnt >= total_num // 2:
                        a = i 
                    if b == -1 and cnt >= total_num // 2 + 1:
                        b = i 
                    if b != -1:
                        break
                else:
                    if cnt >= total_num // 2 + 1:
                        median = i 
                        break
        if a != -1 and b != -1:
            median = (a + b) / 2
        mean = total_val / total_num
        return [mn, mx, mean, median, mode]
```