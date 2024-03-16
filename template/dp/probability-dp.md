### 688. Knight Probability in Chessboard

```python
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        directions = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        @cache
        def dfs(r, c, move):
            if move == k:
                return 1
            if move > k:
                return 0
            res = 0
            for dr, dc in directions:
                rw, cl = r + dr, c + dc 
                if 0 <= rw < R and 0 <= cl < C:
                    res += dfs(rw, cl, move + 1) / 8 
            return res
        R, C = n, n
        return dfs(row, column, 0)
```

### 1230. Toss Strange Coins

```python
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        @cache
        def dfs(i, count):
            if i == n:
                if count == target:
                    return 1
                else:
                    return 0
            if count > target:
                return 0
            return dfs(i + 1, count + 1) * prob[i] + dfs(i + 1, count) * (1 - prob[i])

        n = len(prob)
        return dfs(0, 0)
```