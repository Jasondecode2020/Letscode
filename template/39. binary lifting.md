## template: binary lifting

- dp idea
- use bit manipulation

### 1483. Kth Ancestor of a Tree Node

```python
class TreeAncestor:

    def __init__(self, n: int, parent: List[int]):
        self.p = [[-1] * 32 for i in range(n)]
        for i in range(n):
            self.p[i][0] = parent[i]
        for j in range(1, 32):
            for i in range(n):
                if self.p[i][j - 1] != -1:
                    self.p[i][j] = self.p[self.p[i][j - 1]][j - 1]

    def getKthAncestor(self, node: int, k: int) -> int:
        for j in range(32):
            if k >> j & 1:
                node = self.p[node][j]
                if node == -1:
                    break
        return node
```