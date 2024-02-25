## template 1: Array + ranking

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]
```

## template 2: hash table

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}

    def find(self, n):
        if n != self.parent[n]:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]
    
    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2
```

## template 3: Array + path compression

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, root):
        n = root # use n for path compression
        while self.parent[root] != root:
            root = self.parent[root] # find root first

        while n != root: # start path compression
            nxt = self.parent[n]
            self.parent[n] = root
            n = nxt
        return root # get root

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p2] = p1
```

## template 4: with weight

```python
class UF:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.weight = [1 for i in range(n)]
    
    def find(self, n):
        if n != self.parent[n]:
            origin = self.parent[n]
            self.parent[n] = self.find(self.parent[n])
            self.weight[n] = self.weight[n] * self.weight[origin]
        return self.parent[n]

    def union(self, n1, n2, val):
        p1 = self.find(n1)
        p2 = self.find(n2)
        if p1 != p2:
            self.parent[p1] = p2
            self.weight[p1] = val * self.weight[n2] / self.weight[n1]

    def isConnected(self, n1, n2):
        p1 = self.find(n1)
        p2 = self.find(n2)
        if p1 == p2:
            return self.weight[n1] / self.weight[n2]
        else:
            return -1
```

## template 5: dynamic parent

```python
class UF:
    def __init__(self, nums):
        self.parent = {}
        for x, y in nums:
            self.parent[x] = x
            self.parent[y + 10001] = x

    def find(self, n):
        if n != self.parent[n]:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2
```

### questions

- 684. Redundant Connection
- 547. Number of Provinces
- 990. Satisfiability of Equality Equations
- 200. Number of Islands
- 128. Longest Consecutive Sequence
* [1971. Find if Path Exists in Graph](#1971-Find-if-Path-Exists-in-Graph)
- 323. Number of Connected Components in an Undirected Graph
- 827. Making A Large Island
- 685. Redundant Connection II
- 778. Swim in Rising Water (union find with binary search)
- 765. Couples Holding Hands
- 399. Evaluate Division (union find with weight)
- 1319. Number of Operations to Make Network Connected
- 1631. Path With Minimum Effort (union find with binary search)
- 959. Regions Cut By Slashes (encoding)
- 1202. Smallest String With Swaps
- 947. Most Stones Removed with Same Row or Column (dynamic parent)

### 684. Redundant Connection

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        uf = UF(n)
        for u, v in edges:
            if uf.connected(u - 1, v - 1):
                return [u, v]
            uf.union(u - 1, v - 1)
```

### 547. Number of Provinces

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        uf = UF(n)
        count = n
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1 and not uf.connected(i, j):
                    uf.union(i, j)
                    count -= 1
        return count
```

### 990. Satisfiability of Equality Equations

- hash table

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}

    def find(self, n):
        if n != self.parent[n]:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]
    
    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        uf = UF(string.ascii_lowercase)
        for s in equations:
            if s[1] == '=':
                uf.union(s[0], s[-1])
        
        for s in equations:
            if s[1] == '!' and uf.connected(s[0], s[-1]):
                return False
        return True
```

- array

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        uf = UF(26)
        for s in equations:
            if s[1] == '=':
                uf.union(ord(s[0]) - ord('a'), ord(s[-1]) - ord('a'))
        
        for s in equations:
            if s[1] == '!' and uf.connected(ord(s[0]) - ord('a'), ord(s[-1]) - ord('a')):
                return False
        return True
```

### 200. Number of Islands

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n
    
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        uf = UF(R * C)
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == "1":
                    res += 1
                    for dr, dc in [[1, 0], [0, 1]]:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and grid[row][col] == "1" and uf.find(row * C + col) != uf.find(r * C + c):
                            uf.union(row * C + col, r * C + c)
                            res -= 1
        return res
```

### 128. Longest Consecutive Sequence

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}

    def find(self, n):
        if n != self.parent[n]:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]
    
    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        uf, nums = UF(nums), set(nums)
        for n in nums:
            if n + 1 in nums and not uf.connected(n, n + 1):
                uf.union(n, n + 1)

        d = defaultdict(int)
        for n in nums:
            d[uf.find(n)] += 1
        return max(d.values()) if d.values() else 0
```

### 1971. Find if Path Exists in Graph

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        uf = UF(n)
        for u, v in edges:
            if not uf.connected(u, v):
                uf.union(u, v)
        return uf.connected(source, destination)
```

### 323. Number of Connected Components in an Undirected Graph

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, root):
        n = root # use n for path compression
        while self.parent[root] != root:
            root = self.parent[root] # find root first

        while n != root: # start path compression
            nxt = self.parent[n]
            self.parent[n] = root
            n = nxt
        return root # get root

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p2] = p1

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = UF(n)
        for u, v in edges:
            if not uf.connected(u, v):
                uf.union(u, v)
                n -= 1
        return n
```

### 827. Making A Large Island

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n
    
    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        uf = UF(R * C)
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    for dr, dc in [[1, 0], [0, 1]]:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and grid[row][col] and not uf.connected(row * C + col, r * C + c):
                            uf.union(row * C + col, r * C + c)
        
        res = 0
        for r in range(R):
            for c in range(C):
                cur_uf, ans, visited = uf, 1, set()
                if grid[r][c] == 0:
                    for dr, dc in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and grid[row][col]:
                            root = cur_uf.find(row * C + col)
                            if root not in visited:
                                ans += cur_uf.rank[root]
                                visited.add(root)
                res = max(res, ans, cur_uf.rank[cur_uf.find(r * C + c)])
        return res
```

### 685. Redundant Connection II

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        uf = UF(n + 1)
        parent = uf.parent[::]

        conflict, cycle = -1, -1
        for i, (node1, node2) in enumerate(edges):
            if parent[node2] != node2: # conflict
                conflict = i # conflict edge not connected
            else:
                parent[node2] = node1
                # after produce cycle, conflict edge connected
                if uf.find(node1) == uf.find(node2):
                    cycle = i                   
                else:
                    uf.union(node1, node2) 

        if conflict == -1:                                            
            return [edges[cycle][0], edges[cycle][1]]
        else:               
            conflictEdge = edges[conflict]
            if cycle != -1:
                return [parent[conflictEdge[1]], conflictEdge[1]]
            else:
                return [conflictEdge[0], conflictEdge[1]]     
```

### 778. Swim in Rising Water

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    if r == 0 and c == 0 and grid[r][c] > threshold:
                        return False # speed up a bit
                    if grid[r][c] <= threshold:
                        for dr, dc in [[1, 0], [0, 1]]:
                            row, col = r + dr, c + dc
                            if 0 <= row < R and 0 <= col < C and grid[row][col] <= threshold and not uf.connected(row * C + col, r * C + c):
                                uf.union(row * C + col, r * C + c)
            return uf.connected(0, (R - 1) * C + C - 1)

        R, C = len(grid), len(grid[0])
        l, r, res = 0, R * C, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1102. Path With Maximum Minimum Value

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def maximumMinimumPath(self, grid: List[List[int]]) -> int:
        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    if r == 0 and c == 0 and grid[r][c] < threshold:
                        return False # speed up a bit
                    if grid[r][c] >= threshold:
                        for dr, dc in [[1, 0], [0, 1]]:
                            row, col = r + dr, c + dc
                            if 0 <= row < R and 0 <= col < C and grid[row][col] >= threshold and not uf.connected(row * C + col, r * C + c):
                                uf.union(row * C + col, r * C + c)
            return uf.connected(0, (R - 1) * C + C - 1)

        R, C = len(grid), len(grid[0])
        l, r, res = 0, max(max(item) for item in grid), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                l = m + 1
            else:
                r = m - 1
        return res
```

### 765. Couples Holding Hands

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        n = len(row)
        uf = UF(n // 2) # use couple as single connected node
        for i in range(0, n, 2):
            # couple no. for i and i + 1 are the same
            c1, c2 = row[i] // 2, row[i + 1] // 2
            if c1 != c2:
                uf.union(c1, c2)
        return sum(uf.parent[i] != i for i in range(n // 2))
```

### 399. Evaluate Division

```python
class UF:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.weight = [1 for i in range(n)]
    
    def find(self, n):
        if n != self.parent[n]:
            origin = self.parent[n]
            self.parent[n] = self.find(self.parent[n])
            self.weight[n] = self.weight[n] * self.weight[origin]
        return self.parent[n]

    def union(self, n1, n2, val):
        p1 = self.find(n1)
        p2 = self.find(n2)
        if p1 != p2:
            self.parent[p1] = p2
            self.weight[p1] = val * self.weight[n2] / self.weight[n1]

    def isConnected(self, n1, n2):
        p1 = self.find(n1)
        p2 = self.find(n2)
        if p1 == p2:
            return self.weight[n1] / self.weight[n2]
        else:
            return -1

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        n = len(equations)
        uf, d, id = UF(2 * n), {}, 0
        for (v1, v2), val in zip(equations, values):
            if v1 not in d:
                d[v1] = id
                id += 1
            if v2 not in d:
                d[v2] = id
                id += 1
            uf.union(d[v1], d[v2], val)

        res = [0] * len(queries) 
        for i, (v1, v2) in enumerate(queries):
            if v1 not in d or v2 not in d:
                res[i] = -1
            else:
                res[i] = uf.isConnected(d[v1], d[v2])
        return res
```

1319. Number of Operations to Make Network Connected

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        uf, extra_edges = UF(n), 0
        for u, v in connections:
            if not uf.connected(u, v):
                uf.union(u, v)
            else:
                extra_edges += 1
        d = defaultdict(int)
        for i in range(n):
            d[uf.find(i)] += 1
        components = len(d)
        return components - 1 if extra_edges >= components - 1 else -1
```

### 1631. Path With Minimum Effort

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def minimumEffortPath(self, grid: List[List[int]]) -> int:
        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    for dr, dc in [[1, 0], [0, 1]]:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and abs(grid[row][col] - grid[r][c]) <= threshold and not uf.connected(row * C + col, r * C + c):
                            uf.union(row * C + col, r * C + c)
            return uf.connected(0, (R - 1) * C + C - 1)

        R, C = len(grid), len(grid[0])
        minNum, maxNum = 0, 0
        for row in grid:
            minNum = min(minNum, min(row))
            maxNum = max(maxNum, max(row))
        l, r, res = 0, maxNum - minNum, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 959. Regions Cut By Slashes

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.count = n 

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        if self.connected(n1, n2):
            return
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]
        self.count -= 1

class Solution:
    def regionsBySlashes(self, grid):
        R, C = len(grid), len(grid[0])
        uf = UF(R * C * 4)
        def pos(r, c, i):
            return (r * C + c) * 4 + i
        for r in range(R):
            for c in range(C):
                v = grid[r][c]
                if r + 1 < R: # connect to right
                    uf.union(pos(r, c, 3), pos(r + 1, c, 1))
                if c + 1 < C: # connect to bottom
                    uf.union(pos(r, c, 2), pos(r, c + 1, 0))
                if v == '/': # depend on encoding of 1, 2, 3, 4
                    uf.union(pos(r, c, 0), pos(r, c, 1))
                    uf.union(pos(r, c, 2), pos(r, c, 3))
                if v == '\\': # escape '\\' means '\'
                    uf.union(pos(r, c, 0), pos(r, c, 3))
                    uf.union(pos(r, c, 1), pos(r, c, 2))
                if v == ' ': # connect all
                    uf.union(pos(r, c, 0), pos(r, c, 1))
                    uf.union(pos(r, c, 1), pos(r, c, 2))
                    uf.union(pos(r, c, 2), pos(r, c, 3))
        return uf.count
```

1202. Smallest String With Swaps

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        uf = UF(n)
        for u, v in pairs:
            if not uf.connected(u, v):
                uf.union(u, v)

        d = defaultdict(list)
        for i in range(n):
            d[uf.find(i)].append(i)
            
        s = list(s)
        for arr in d.values():
            sortedLetters = sorted([s[i] for i in arr])
            for i in range(len(sortedLetters)):
                s[arr[i]] = sortedLetters[i]
        return ''.join(s)
```

947. Most Stones Removed with Same Row or Column

```python
class UF:
    def __init__(self, nums):
        self.parent = {}
        for x, y in nums:
            self.parent[x] = x
            self.parent[y + 10001] = x

    def find(self, n):
        if n != self.parent[n]:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        uf = UF(stones)
        for x, y in stones:
            if not uf.connected(x, y + 10001):
                uf.union(x, y + 10001)
        return len(stones) - sum(1 for k, v in uf.parent.items() if k == v)
```

### 2316. Count Unreachable Pairs of Nodes in an Undirected Graph

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        uf = UF(n)
        for u, v in edges:
            if not uf.connected(u, v):
                uf.union(u, v)

        d = defaultdict(int)
        for i in range(n):
            d[uf.find(i)] += 1
        nums = list(d.values())
        presum = list(accumulate(nums, initial = 0))
        res, n = 0, len(nums)
        for i in range(n - 1):
            res += nums[i] * (presum[-1] - presum[i + 1])
        return res
```

### 1559. Detect Cycles in 2D Grid

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        uf = UF(R * C)
        for r in range(R):
            for c in range(C):
                for dr, dc in [[1, 0], [0, 1]]:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C and grid[row][col] == grid[r][c]:
                        if uf.connected(row * C + col, r * C + c):
                            return True
                        uf.union(row * C + col, r * C + c)
        return False
```

- simple template

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        if n != self.parent[n]:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]
    
    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        uf = UF(R * C)
        for r in range(R):
            for c in range(C):
                for dr, dc in [[1, 0], [0, 1]]:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C and grid[row][col] == grid[r][c]:
                        if uf.connected(row * C + col, r * C + c):
                            return True
                        uf.union(row * C + col, r * C + c)
        return False
```

- dfs

```python
class Solution:
    def containsCycle(self, grid: List[List[str]]) -> bool:
        sys.setrecursionlimit(150000)
        R, C = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited = set()
        def dfs(node, p):
            if node in visited: 
                return True
            visited.add(node)
            r, c = node
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) != p and grid[row][col] == grid[r][c]:
                    if dfs((row, col), node): 
                        return True
            return False
        
        for r in range(R):
            for c in range(C):
                if (r, c) not in visited and dfs((r, c), None):
                    return True
        return False
```

### 1091. Shortest Path in Binary Matrix

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        if grid[0][0]:
            return -1
        q = deque([(0, 0, 0)])
        visited = set([(0, 0)])
        while q:
            r, c, d = q.popleft()
            if r == R - 1 and c == C - 1:
                return d + 1
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited and grid[row][col] == 0:
                    visited.add((row, col))
                    q.append((row, col, d + 1))
        return -1
```

