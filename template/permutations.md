* [2850. Minimum Moves to Spread Stones Over Grid](#2850-minimum-moves-to-spread-stones-over-grid)

### 2850. Minimum Moves to Spread Stones Over Grid

```python
class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        move, receive = [], []
        res = inf 
        for r in range(R):
            for c in range(C):
                if grid[r][c] > 1:
                    move.extend([(r, c)] * (grid[r][c] - 1))
                elif grid[r][c] == 0:
                    receive.append((r, c))

        for m in permutations(move):
            ans = 0
            for (x1, y1), (x2, y2) in zip(m, receive):
                ans += abs(x1 - x2) + abs(y1 - y2)
            res = min(res, ans)
        return res
```