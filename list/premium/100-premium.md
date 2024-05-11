## Merge intervals (2)

* [758. Bold Words in String](#758-bold-words-in-string)
* [616. Add Bold Tag in String](#616-add-bold-tag-in-string)

## Graph BFS (1)

* [1197. Minimum Knight Moves](#1197-minimum-knight-moves)


### 616. Add Bold Tag in String
### 758. Bold Words in String (same as 616)

```python
class Solution:
    def addBoldTag(self, s: str, words: List[str]) -> str:
        n = len(s)
        intervals = [] 
        words = set(words)
        for i in range(n):
            for w in words:
                if s[i:].startswith(w):
                    if not intervals or intervals[-1][1] < i:
                        intervals.append([i, i + len(w)])
                    else:
                        intervals[-1][1] = max(intervals[-1][1], i + len(w))

        res = ''
        prev = 0
        if not intervals:
            return s
        for a, b in intervals:
            res += s[prev: a] + '<b>' + s[a: b] + '</b>'
            prev = b 
        return res + s[b:]
```
## BFS 

* [1197. Minimum Knight Moves](#1197-minimum-knight-moves)

### 1197. Minimum Knight Moves

```python
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        directions = [[2, 1], [1, 2], [-1, 2], [-2, 1], [-2, -1], [-1, -2], [1, -2], [2, -1]]
        q = deque([(0, 0, 0)])
        visited = set([(0, 0)])
        while q:
            r, c, step = q.popleft()
            if r == x and c == y:
                return step
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if (row, col) not in visited:
                    visited.add((row, col))
                    q.append((row, col, step + 1))
```

### 254. Factor Combinations

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        def backtrack(n, ans):
            if len(ans) > 0:
                res.append(ans + [n])
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    if not ans or i >= ans[-1]:
                        backtrack(n // i, ans + [i])

        res = []
        backtrack(n, [])
        return res 
```