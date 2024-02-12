### 847. Shortest Path Visiting All Nodes

```python
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        q, visited = deque(), set()
        end = (1 << n) - 1
        for i in range(n):
            q.append((i, 1 << i, 0))
            visited.add((i, 1 << i))
        while q:
            node, state, step = q.popleft()
            if state == end:
                return step
            for nei in graph[node]:
                nei_state = state | (1 << nei) 
                if (nei, nei_state) not in visited:
                    q.append((nei, nei_state, step + 1))
                    visited.add((nei, nei_state))
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