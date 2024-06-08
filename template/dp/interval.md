## job schedule dp

### 2008. Maximum Earnings From Taxi

```python
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        f, d = [0] * (n + 1), defaultdict(list) # max profit at i
        for s, e, t in rides:
            d[e].append((s, t))

        for e in range(1, n + 1):
            f[e] = f[e - 1] # profit for not choosing e
            if e in d: # choose e
                for s2, t2 in d[e]:
                    f[e] = max(f[e], e - s2 + t2 + f[s2])      
        return f[-1]
```