### 492. Construct the Rectangle

```python
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        ans = inf 
        res = []
        for i in range(1, int(sqrt(area)) + 1):
            if area % i == 0 and abs(i - area // i) < ans:
                res = [area // i, i]
        return res
```

### 1390. Four Divisors

```python
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def check(n):
            res = set()
            for i in range(1, int(sqrt(n)) + 1):
                if n % i == 0:
                    res.add(i)
                    res.add(n // i)
                    if len(res) > 4:
                        break
            if len(res) == 4:
                return sum(list(res))
            return 0

        res = 0
        for n in nums:
            res += check(n)
        return res
```

### 1362. Closest Divisors

```python
class Solution:
    def closestDivisors(self, num: int) -> List[int]:
        n1, n2 = num + 1, num + 2
        res = [-inf, inf]
        for i in range(1, int(sqrt(n2)) + 1):
            if n1 % i == 0:
                a, b = i, n1 // i
                if b - a < res[1] - res[0]:
                    res = [a, b]
            if n2 % i == 0:
                a, b = i, n2 // i
                if b - a < res[1] - res[0]:
                    res = [a, b]
        return res
```