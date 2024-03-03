## change root dp

- use dfs for one root
- change root using another dfs

### 2581. Count Number of Possible Root Nodes

```python
class Solution:
    def rootCount(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        guesses = set([(u, v) for u, v in guesses])

        self.count = 0
        def dfs(node, parent):
            for nei in g[node]:
                if nei != parent:
                    if (node, nei) in guesses:
                        self.count += 1
                    dfs(nei, node)
        dfs(0, -1)

        self.res = 0
        def change_root(node, parent, count):
            if count >= k:
                self.res += 1
            for nei in g[node]:
                if nei != parent:
                    change_root(nei, node, count - ((node, nei) in guesses) + ((nei, node) in guesses))
        change_root(0, -1, self.count)
        return self.res
```

### 834. Sum of Distances in Tree

```python
class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        res = [0] * n 
        size = [1] * n 
        def dfs(x, fa, depth):
            res[0] += depth 
            for y in g[x]:
                if y != fa:
                    dfs(y, x, depth + 1)
                    size[x] += size[y]
        dfs(0, -1, 0)

        def shiftroot(x, fa):
            for y in g[x]:
                if y != fa:
                    res[y] = res[x] + n - 2 * size[y]
                    shiftroot(y, x)
        shiftroot(0, -1)
        return res
```