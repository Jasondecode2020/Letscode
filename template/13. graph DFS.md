## template 1: recursion

```python
def fn(g): # g: graph
    def dfs(node):
        res = 0
        # some code
        for neighbor in g[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                ans += dfs(neighbor)
        return res

    visited = {START_NODE}
    return dfs(START_NODE)
```

### template 2: iteration

```python
def fn(g): # g: graph
    stack = [START_NODE]
    visited = {START_NODE}
    res = 0
    while stack:
        node = stack.pop()
        # some code
        for neighbor in g[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    
    return res
```

## template 3: grid

```python
def fn(grid): # grid: 695
    R, C = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r == R or c < 0 or c == C or grid[r][c] == 0:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
    return max(dfs(r, c) for r in range(R) for c in range(C))
```


### 695. Max Area of Island

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        def dfs(r, c):
            if r < 0 or r == R or c < 0 or c == C or grid[r][c] == 0:
                return 0
            grid[r][c] = 0
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
        return max(dfs(r, c) for r in range(R) for c in range(C))
```

### 200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        def dfs(r, c):
            if r < 0 or r == R or c < 0 or c == C or grid[r][c] == '0':
                return
            grid[r][c] = '0'
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
        
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    dfs(r, c)
                    res += 1
        return res
```

### 1905. Count Sub Islands

```python
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        def dfs(r, c):
            if r < 0 or c < 0 or r == R or c == C or grid2[r][c] == 0:
                return
            grid2[r][c] = 0
            if grid1[r][c] == 0:
                self.isSub = False
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        R, C = len(grid2), len(grid2[0])
        res = 0
        for r in range(R):
            for c in range(C):
                self.isSub = True
                if grid2[r][c] == 1:
                    dfs(r, c)
                    if self.isSub: res += 1
                    
        return res
```

### 2246. Longest Path With Different Adjacent Characters

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        g = defaultdict(list)
        n = len(parent)
        for i in range(1, n):
            g[parent[i]].append(i)

        self.res = 0
        def dfs(x):
            x_len = 0
            for y in g[x]:
                y_len = dfs(y) + 1
                if s[y] != s[x]:
                    # longest and second longest
                    self.res = max(self.res, x_len + y_len)
                    x_len = max(x_len, y_len)
            return x_len
        dfs(0)
        return self.res + 1
```

### 1245. Tree Diameter

```python
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        self.res = 0
        def dfs(x, parent): # dfs(0), x = 0, parent = -1, dfs(1, 0)
            first = 0
            for y in g[x]:
                if y == parent:
                    continue
                second = dfs(y, x) + 1
                self.res = max(self.res, first + second)
                first = max(first, second)
            return first
        dfs(0, -1)
        return self.res
```