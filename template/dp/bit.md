### Question list

### bit dp (6)

* [526. Beautiful Arrangement](#526-beautiful-arrangement)
* [2741. Special Permutations](#2741-special-permutations)
* [698. Partition to K Equal Sum Subsets](#698-Partition-to-K-Equal-Sum-Subsets)
* [847. Shortest Path Visiting All Nodes](#847-Shortest-Path-Visiting-All-Nodes)
* [1284. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix](#1284-minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix)
* [864. Shortest Path to Get All Keys](#864-shortest-path-to-get-all-keys)

### 526. Beautiful Arrangement

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        state = (1 << n) - 1
        print(state)
        @cache
        def dfs(s):
            if s == state:
                return 1
            i = s.bit_count() 
            res = 0
            for j in range(1, n + 1):
                if s & 1 << (j - 1) == 0 and (i % j == 0 or j % i == 0):
                    res += dfs(s | 1 << (j - 1))
            return res 
        return dfs(0)
```

### 2741. Special Permutations

```python
class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        mod = 10 ** 9 + 7
        n = len(nums)
        state = (1 << n) - 1
        @cache
        def dfs(s, prev):
            if s == state:
                return 1
            res = 0
            for i, num in enumerate(nums):
                if s & 1 << i == 0 and (prev == -1 or num % prev == 0 or prev % num == 0):
                    res += dfs(s | 1 << i, num)
            return res 
        return dfs(0, -1) % mod
```

### 698. Partition to K Equal Sum Subsets

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        @cache   
        def dfs(state, total):
            if state == (1 << n) - 1:
                return True
            for j in range(n):
                # if total + nums[j] > target:# nums sorted
                #     break
                if state & (1 << j) == 0 and total + nums[j] <= target:     
                    next_state = state | (1 << j)     
                    res = total + nums[j] if total + nums[j] < target else 0
                    if dfs(next_state, res):    
                        return True
            return False
        
        total = sum(nums)
        if total % k != 0:
            return False
        n = len(nums)
        target = total // k   
        nums.sort()   
        return dfs(0, 0)

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        @cache   
        def dfs(state, total):
            if state == (1 << n) - 1:
                return True
            for j in range(n):
                if total + nums[j] > target:
                    break
                if state & (1 << j) == 0:     
                    next_state = state | (1 << j)     
                    res = total + nums[j] if total + nums[j] < target else 0
                    if dfs(next_state, res):    
                        return True
            return False
        
        total = sum(nums)
        if total % k != 0:
            return False
        n = len(nums)
        target = total // k   
        nums.sort()   
        return dfs(0, 0)
```

### 847. Shortest Path Visiting All Nodes

```python
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        end = (1 << n) - 1
        q, visited = deque(), set()
        for i in range(n):
            q.append((i, 1 << i, 0))
            visited.add((i, 1 << i)) # cur_node, visited_nodes
        while q:
            cur, state, steps = q.popleft()
            if state == end:
                return steps 
            for nei in graph[cur]:
                new_state = state | (1 << nei)
                if (nei, new_state) not in visited:
                    q.append((nei, new_state, steps + 1))
                    visited.add((nei, new_state))
```

### 1284. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix

```python
class Solution:
    def minFlips(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        start = ''
        for row in mat:
            for col in row:
                start += str(col)
        start = int(start, 2)
        visited = set([start])
        q = deque([(start, 0)])
        directions = [[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]]
        while q:
            state, steps = q.popleft()
            if state == 0:
                return steps
            for r in range(R):
                for c in range(C):
                    nxt_state = state
                    for dr, dc in directions:
                        row, col = r + dr, c + dc 
                        if 0 <= row < R and 0 <= col < C:
                            nxt_state = nxt_state ^ (1 << (R * C - row * C - col - 1))
                    if nxt_state not in visited:
                        visited.add(nxt_state)
                        q.append((nxt_state, steps + 1))
        return -1
```

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

