### 296. Best Meeting Point

```python
class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        r_sum, c_sum = [], []
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    r_sum.append(r)
                    c_sum.append(c)
        r_sum.sort()
        c_sum.sort()

        r_mid, c_mid = r_sum[len(r_sum) // 2], c_sum[len(c_sum) // 2]
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    res += abs(r - r_mid) + abs(c - c_mid)
        return res
```
