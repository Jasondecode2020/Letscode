### Question list

1 [698. Partition to K Equal Sum Subsets](#698-Partition-to-K-Equal-Sum-Subsets)
2 [847. Shortest Path Visiting All Nodes](#847-Shortest-Path-Visiting-All-Nodes)

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

### 698. Partition to K Equal Sum Subsets

```python
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
                    next_state = state + (1 << j)     
                    res = total + nums[j] if (total + nums[j]) % target != 0 else 0
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

### 465. Optimal Account Balancing

```python
class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        person = defaultdict(int)
        for u, v, c in transactions:
            person[u] -= c
            person[v] += c
        accounts = list(person.values())
       
        self.res = inf
        n = len(accounts)
        def dfs(i, cnt):
            if cnt >= self.res: return 
            while i < n and accounts[i] == 0: 
                i += 1
            if i == n:
                self.res = min(self.res, cnt)
                return
              
            for j in range(i + 1, n):
                if accounts[i] * accounts[j] < 0:
                    accounts[j] += accounts[i]
                    dfs(i + 1, cnt + 1)
                    accounts[j] -= accounts[i]
        dfs(0, 0)
        return self.res
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