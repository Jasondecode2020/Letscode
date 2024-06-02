### 864. Shortest Path to Get All Keys

```python
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        R, C, directions = len(grid), len(grid[0]), [[0, 1], [0, -1], [1, 0], [-1, 0]]
        x, y = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == '@'].pop()
        n = sum(1 for r in range(R) for c in range(C) if grid[r][c].islower())
        q, visited = deque([(x, y, 0)]), set([(x, y, 0)])
        res = 0
        while q:
            for _ in range(len(q)):
                x, y, state = q.popleft()
                if state == (1 << n) - 1:
                    return res 
                for dx, dy in directions:
                    row, col = x + dx, y + dy
                    nxt = state
                    if 0 <= row < R and 0 <= col < C:
                        c = grid[row][col]
                        if c == '#' or c.isupper() and (state & (1 << (ord(c) - ord('A')))) == 0:
                            continue
                        if c.islower():
                            nxt |= 1 << (ord(c) - ord('a'))
                        if (row, col, nxt) not in visited:
                            visited.add((row, col, nxt))
                            q.append((row, col, nxt))
            res += 1
        return -1
```