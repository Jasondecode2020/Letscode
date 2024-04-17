## template 1: Array + ranking

- continous uf with rank

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

    def isConnected(self, n1, n2):
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

## template 2: hash table: leetcode 128

- discrete uf with rank

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}
        self.rank = {n: 1 for n in nums}
    
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

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)
```

## template 3: with weight

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

## template 4: dynamic parent

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

### Questions list

* 1 [128. Longest Consecutive Sequence](#128-Longest-Consecutive-Sequence)
* 2 [130. Surrounded Regions](#130-Surrounded-Regions)
* 3 [200. Number of Islands](#200-Number-of-Islands)
* 4 [261. Graph Valid Tree](#261-Graph-Valid-Tree)
* 5 [305. Number of Islands II](#305-Number-of-Islands-II)

* 6 [323. Number of Connected Components in an Undirected Graph](#323-Number-of-Connected-Components-in-an-Undirected-Graph)
* 7 [399. Evaluate Division](#399-Evaluate-Division)
* 8 [547. Number of Provinces](#547-Number-of-Provinces)
* 9 [684. Redundant Connection](#684-Redundant-Connection)
* 10 [685. Redundant Connection II](#685-Redundant-Connection-II)

* 11 [694. Number of Distinct Islands](#694-Number-of-Distinct-Islands)
* 12 [695. Max Area of Island](#695-Max-Area-of-Island)
* 13 [711. Number of Distinct Islands II](#711-Number-of-Distinct-Islands-II)
* 14 [721. Accounts Merge](#721-Accounts-Merge)
* 15 [737. Sentence Similarity II](#737-Sentence-Similarity-II)


* 16 [765. Couples Holding Hands](#765-Couples-Holding-Hands)
* 17 [778. Swim in Rising Water](#778-Swim-in-Rising-Water)
* 18 [785. Is Graph Bipartite?](#785-Is-Graph-Bipartite?)
* 19 [827. Making A Large Island](#827-Making-A-Large-Island)
* 20 [839. Similar String Groups](#839-Similar-String-Groups)

* [886. Possible Bipartition](#886-Possible-Bipartition)

* 21 [947. Most Stones Removed with Same Row or Column](#947-Most-Stones-Removed-with-Same-Row-or-Column)
* 22 [959. Regions Cut By Slashes](#959-Regions-Cut-By-Slashes)
* 23 [990. Satisfiability of Equality Equations](#990-Satisfiability-of-Equality-Equations)
* 24 [1061. Lexicographically Smallest Equivalent String](#1061-Lexicographically-Smallest-Equivalent-String)
* 25 [1101. The Earliest Moment When Everyone Become Friends](#1101-The-Earliest-Moment-When-Everyone-Become-Friends)

* 21 [1722. Minimize Hamming Distance After Swap Operations](#1722-Minimize-Hamming-Distance-After-Swap-Operations)
* 22 [1202. Smallest String With Swaps](#1202-Smallest-String-With-Swaps)
* 23 [1319. Number of Operations to Make Network Connected](#1319-Number-of-Operations-to-Make-Network-Connected)
* 24 [1631. Path With Minimum Effort](#1631-Path-With-Minimum-Effort)
* 25 [1970. Last Day Where You Can Still Cross](#1970-Last-Day-Where-You-Can-Still-Cross)

* 26 [1971. Find if Path Exists in Graph](#1971-Find-if-Path-Exists-in-Graph)
* 27 [2092. Find All People With Secret](#2092-Find-All-People-With-Secret)










### 128. Longest Consecutive Sequence

- hash set

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        res = 0
        for n in nums:
            if n - 1 not in nums:
                ans = 1
                while n + 1 in nums:
                    ans += 1
                    n += 1
                res = max(res, ans)
        return res
```

- discrete mathematics

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.parent[n1], self.parent[n2]
        self.parent[p1] = p2

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        uf = UF(nums)
        nums = set(nums)
        for n in nums:
            if n + 1 in nums:
                uf.union(n, n + 1)

        d = Counter()
        for n in uf.parent:
            d[uf.find(n)] += 1
        res = list(d.values()) + [0]
        return max(res)
```

- with rank

```python
class UF:
    def __init__(self, nums):
        self.parent = {n: n for n in nums}
        self.rank = {n: 1 for n in nums}
    
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
    def longestConsecutive(self, nums: List[int]) -> int:
        uf = UF(nums)
        nums = set(nums)
        for n in nums:
            if n + 1 in nums:
                uf.union(n, n + 1)

        d = Counter()
        for n in uf.parent:
            d[uf.find(n)] += 1
        res = list(d.values()) + [0]
        return max(res)
```

### 130. Surrounded Regions

- DFS

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        R, C = len(board), len(board[0])
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        def dfs(r, c):
            board[r][c] = '#'
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and board[row][col] == 'O':
                    dfs(row, col)
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and board[r][c] == 'O':
                    dfs(r, c)
        for r in range(R):
            for c in range(C):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
        for r in range(R):
            for c in range(C):
                if board[r][c] == '#':
                    board[r][c] = 'O'
```

- BFS

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        R, C = len(board), len(board[0])
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        def bfs():
            q = deque()
            for r in range(R):
                for c in range(C):
                    if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and board[r][c] == 'O':
                        q.append((r, c))
            while q:
                r, c = q.popleft()
                board[r][c] = '#'
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and board[row][col] == 'O':
                        q.append((row, col))
        bfs()
        for r in range(R):
            for c in range(C):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
        for r in range(R):
            for c in range(C):
                if board[r][c] == '#':
                    board[r][c] = 'O'
```

- Union Find + dummy point in grid

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

    def isConnected(self, n1, n2):
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
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        R, C = len(board), len(board[0])
        directions = [(0,1),(1,0)]
        uf = UF(R * C + 1)
        dummy = R * C # dummy point
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and board[r][c] == 'O':
                    uf.union(r * C + c, dummy)
        for r in range(R):
            for c in range(C):
                if board[r][c] == 'O':
                    for dr, dc in directions:
                        row, col = r + dr, c + dc 
                        if 0 <= row < R and 0 <= col < C and board[row][col] == 'O':
                            uf.union(r * C + c, row * C + col)
        for r in range(R):
            for c in range(C):
                if not uf.isConnected(r * C + c, dummy):
                    board[r][c] = 'X'
```

### 200. Number of Islands

- DFS

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(r, c):
            grid[r][c] = '0'
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and grid[row][col] == '1':
                    dfs(row, col)

        R, C, res = len(grid), len(grid[0]), 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == '1':
                    dfs(r, c)
                    res += 1
        return res
```

- union find in grid

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

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [[1, 0], [0, 1]]
        uf = UF(R * C)
        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == "1":
                    res += 1
                    for dr, dc in directions:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and grid[row][col] == "1" and not uf.isConnected(row * C + col, r * C + c):
                            uf.union(row * C + col, r * C + c)
                            res -= 1
        return res
```

### 261. Graph Valid Tree

- union find + tree and graph

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if n != len(edges) + 1:
            return False
        uf = UF(n)
        for u, v in edges:
            if uf.isConnected(u, v):
                return False
            uf.union(u, v)
        return True
```

### 305. Number of Islands II

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        uf = UF(m * n)
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        s = set()
        res = 0
        ans = []
        for r, c in positions:
            if (r, c) not in s:
                s.add((r, c))
                res += 1
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < m and 0 <= col < n and (row, col) in s and not uf.isConnected(r * n + c, row * n + col):
                        uf.union(r * n + c, row * n + col)
                        res -= 1
            ans.append(res)
        return ans
```

### 323. Number of Connected Components in an Undirected Graph

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

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

### 399. Evaluate Division

- BFS

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        g = defaultdict(list)
        for (a, b), v in zip(equations, values):
            g[a].append((b, v))
            g[b].append((a, 1.0 / v))

        def bfs(start, end):
            if start not in g or end not in g:
                return -1.0
            q, visited = deque([(start, 1.0)]), set([start])
            while q:
                node, v = q.popleft()
                if node == end:
                    return v 
                for nei, cost in g[node]:
                    if nei not in visited:
                        visited.add(nei)
                        q.append((nei, v * cost))
            return -1
        return [bfs(s, e) for s, e in queries]
```

- union find with weight

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

### 547. Number of Provinces

- DFS

``` python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(i):
            for j in range(n):
                if isConnected[i][j] and j not in visited:
                    visited.add(j)
                    dfs(j)
                    
        res, visited, n = 0, set(), len(isConnected)
        for i in range(n):
            if i not in visited:
                dfs(i)
                res += 1
        return res
```

- union find 

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
        def check(edges, remove):
            edges = [[u, v] for u, v in edges if [u, v] != remove]
            for u, v in edges:
                if uf.connected(u, v):
                    return False
                uf.union(u, v)
            return True

        n = len(edges)
        uf = UF(n + 1)
        indegree = [0] * (n + 1)
        res = []
        for u, v in edges:
            indegree[v] += 1
            if indegree[v] == 2:
                res = [u, v]
        mx = max(indegree)
        if mx < 2: # cycle
            for u, v in edges:
                if uf.connected(u, v):
                    return [u, v]
                uf.union(u, v)
        else: # indegree == 2
            if check(edges, res):
                return res # second edge
            else:
                for e in edges:
                    if e[1] == res[1]:
                        return e # first edge  
```

### 694. Number of Distinct Islands

```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 0
            for i, (dr, dc) in enumerate(directions):
                row, col = r + dr, c + dc
                self.s += str(i)
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    dfs(row, col)

        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        visited = set()
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    self.s = ''
                    dfs(r, c)
                    visited.add(self.s)
        return len(visited)
```

### 695. Max Area of Island

-DFS

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 0
            res = 1
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    res += dfs(row, col)
            return res

        R, C, res = len(grid), len(grid[0]), 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    res = max(res, dfs(r, c))
        return res
```

- union find use rank

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
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [1, 0]]
        uf = UF(R * C)
        hasIsland = False
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    hasIsland = True
                    for dr, dc in directions:
                        row, col = r + dr, c + dc 
                        if 0 <= row < R and 0 <= col < C and grid[row][col] == 1 and not uf.connected(r * C + c, row * C + col):
                            uf.union(r * C + c, row * C + col)
        
        if not hasIsland:
            return 0
        return max(uf.rank)
```

### 711. Number of Distinct Islands II

```python
class Solution:
    def numDistinctIslands2(self, grid: List[List[int]]) -> int:
        def dfs(r, c):
            seen.add((r, c))
            grid[r][c] = 0
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C and grid[row][col] == 1:
                    dfs(row, col)

        def check():
            arr = list(seen)
            n = len(arr)
            dist = 0
            for i in range(n):
                x1, y1 = arr[i]
                for j in range(i + 1, n):
                    x2, y2 = arr[j]
                    dist += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            for v in visited:
                if abs(dist - v) < 1e-5:
                    break
            else:
                visited.add(dist)

        R, C = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        visited = set()
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    seen = set()
                    dfs(r, c)
                    check()
        return len(visited)
```

### 721. Accounts Merge

- dict to union find 

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

    def isConnected(self, n1, n2):
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
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        emailToIndex, emailToName = {}, {}
        for account in accounts:
            name = account[0]
            for email in account[1:]:
                if email not in emailToIndex:
                    emailToIndex[email] = len(emailToIndex)
                    emailToName[email] = name 
        
        uf = UF(len(emailToIndex))
        for account in accounts:
            firstIndex = emailToIndex[account[1]]
            for email in account[2:]:
                uf.union(firstIndex, emailToIndex[email])
        
        d = defaultdict(list)
        for e, i in emailToIndex.items():
            d[uf.find(i)].append(e)
        
        res = []
        for emails in d.values():
            res.append([emailToName[emails[0]]] + sorted(emails))
        return res
```

### 737. Sentence Similarity II

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

    def isConnected(self, n1, n2):
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
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        wordToIndex = {}
        for a, b in similarPairs:
            if a not in wordToIndex:
                wordToIndex[a] = len(wordToIndex)
            if b not in wordToIndex:
                wordToIndex[b] = len(wordToIndex)
        for s1 in sentence1:
            if s1 not in wordToIndex:
                wordToIndex[s1] = len(wordToIndex)
        for s2 in sentence2:
            if s2 not in wordToIndex:
                wordToIndex[s2] = len(wordToIndex)

        uf = UF(len(wordToIndex))
        for a, b in similarPairs:
            index1, index2 = wordToIndex[a], wordToIndex[b]
            uf.union(index1, index2)
        
        if len(sentence1) != len(sentence2):
            return False
        for s1, s2 in zip(sentence1, sentence2):
            index1, index2 = wordToIndex[s1], wordToIndex[s2]
            if not uf.isConnected(index1, index2):
                return False
        return True
```

### 839. Similar String Groups

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            print(n)
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n
    
    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        def check(s1, s2):
            count = 0
            for a, b in zip(s1, s2):
                if a != b:
                    count += 1
                    if count > 2:
                        return False
            return True
            
        n = len(strs)
        uf = UF(n)
        for i in range(n):
            for j in range(i + 1, n):
                if check(strs[i], strs[j]):
                    uf.union(i, j)
        d = Counter()
        for s in uf.parent:
            d[uf.find(s)] += 1
        return len(d)
```

### 886. Possible Bipartition

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
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = defaultdict(list)
        for u, v in dislikes:
            g[u - 1].append(v - 1)
            g[v - 1].append(u - 1)

        uf = UF(n)
        for x, nodes in g.items():
            for y in nodes:
                uf.union(nodes[0], y)
                if uf.connected(x, y):
                    return False
        return True
```

### 990. Satisfiability of Equality Equations

- discrete union find

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
        uf = UF(ascii_lowercase)
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

### 1722. Minimize Hamming Distance After Swap Operations

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            print(n)
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n
    
    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2

class Solution:
    def minimumHammingDistance(self, source: List[int], target: List[int], allowedSwaps: List[List[int]]) -> int:
        uf = UF(len(source))
        for u, v in allowedSwaps:
            if not uf.isConnected(u, v):
                uf.union(u, v)
        d = defaultdict(list)
        for n in range(len(source)):
            d[uf.find(n)].append(n)
        res = 0
        for v in d.values():
            c1, c2 = Counter(), Counter()
            count = 0
            for i in v:
                c1[source[i]] += 1
                c2[target[i]] += 1
            for n in c1:
                if n not in c2:
                    count += c1[n]
                elif c2[n] < c1[n]:
                    count += c1[n] - c2[n]
            res += count
        return res
```

### 1971. Find if Path Exists in Graph

- DFS

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def dfs(source):
            if source == destination:
                self.res = True
                return 
            for nei in g[source]:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)

        visited = set([source])
        self.res = False
        dfs(source)
        return self.res
```

- union find 

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

### 785. Is Graph Bipartite?

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
    def isBipartite(self, graph: List[List[int]]) -> bool:
        uf = UF(len(graph))
        for x, nodes in enumerate(graph):
            for y in nodes:
                uf.union(nodes[0], y)
                if uf.connected(x, y):
                    return False
        return True
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
        uf = UF(n // 2)
        for i in range(0, n, 2):
            a, b = row[i] // 2, row[i + 1] // 2
            if a != b:
                uf.union(a, b)
        return sum(uf.find(i) != i for i in range(n // 2))
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

### 2092. Find All People With Secret

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        n1, n2 = self.find(n1), self.find(n2)
        self.parent[n1] = n2 

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def disconnect(self, n):
        self.parent[n] = n 

class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        uf = UF(n)
        uf.union(0, firstPerson)
        people = set()
        meetings.sort(key = lambda x: x[2])
        time = 0
        for u, v, t in meetings:
            if t != time:
                time = t
                for p in people:
                    if not uf.isConnected(p, 0):
                        uf.disconnect(p)
                people = set()
            uf.union(u, v)
            people.add(u)
            people.add(v)
        
        res = set()
        for i in range(n):
            if uf.isConnected(i, 0):
                res.add(i)
        return list(res)
```

### 1361. Validate Binary Tree Nodes

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

    def isConnected(self, n1, n2):
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
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        edges = 0
        indegree = [0 for i in range(n)]
        uf = UF(n)
        for i, (l, r) in enumerate(zip(leftChild, rightChild)):
            if l != -1:
                edges += 1
                indegree[l] += 1
                uf.union(i, l)
            if r != -1:
                edges += 1
                indegree[r] += 1
                uf.union(i, r)
        d = Counter()
        for v in uf.parent:
            d[uf.find(v)] += 1
        return edges == n - 1 and max(indegree) < 2 and len(d) == 1
```

### 1970. Last Day Where You Can Still Cross

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

    def isConnected(self, n1, n2):
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
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        R, C = row, col 
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        cells = [(u - 1, v - 1) for u, v in cells]
        uf = UF(R * C + 2)
        n = len(cells)
        set_cells = set(cells)
        ceil, floor = R * C, R * C + 1
        for i in range(n - 1, -1, -1):
            r, c = cells[i]
            set_cells.remove((r, c))
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in set_cells and not uf.isConnected(r * C + c, row * C + col):
                    uf.union(r * C + c, row * C + col)
            if r + 1 == R:
                uf.union(r * C + c, floor)
            if r == 0:
                uf.union(r * C + c, ceil)
            if uf.isConnected(floor, ceil):
                return i
```

### 1061. Lexicographically Smallest Equivalent String

```python
class UF:
    def __init__(self, nums):
        self.parent = {c: c for c in nums}

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        self.parent[p1] = p2  

class Solution:
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        uf = UF(ascii_lowercase)
        for a, b in zip(s1, s2):
            if not uf.isConnected(a, b):
                uf.union(a, b)

        res = ''
        for c in baseStr:
            for l in ascii_lowercase:
                if uf.isConnected(c, l):
                    res += l 
                    break
        return res
```

### 1101. The Earliest Moment When Everyone Become Friends

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

    def isConnected(self, n1, n2):
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
    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        uf = UF(n)
        logs.sort()
        for t, u, v in logs:
            if not uf.isConnected(u, v):
                uf.union(u, v)
            if max(uf.rank) == n:
                return t 
        return -1
```