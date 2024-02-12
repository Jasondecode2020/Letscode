### 372. Super Pow

```python
class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        res = [0] * len(queries)
        count = [0] * (n + 1)
        out_of_range = n 
        l, r = 0, 0
        logs.sort(key = lambda x: x[1])
        for i, q in sorted(enumerate(queries), key = lambda q: q[1]):
            while r < len(logs) and logs[r][1] <= q:
                Id = logs[r][0]
                if count[Id] == 0:
                    out_of_range -= 1
                count[Id] += 1
                r += 1
            while l < len(logs) and logs[l][1] < q - x:
                Id = logs[l][0]
                count[Id] -= 1
                if count[Id] == 0:
                    out_of_range += 1
                l += 1
            res[i] = out_of_range
        return res
```