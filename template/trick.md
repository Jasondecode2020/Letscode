### 2017. Grid Game

```python
class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        pre = []
        for row in grid:
            pre.append(list(accumulate(row, initial = 0)))
        res = inf
        for i in range(1, len(pre[0])):
            res = min(res, max(pre[0][-1] - pre[0][i], pre[1][i - 1]))
        return res
```