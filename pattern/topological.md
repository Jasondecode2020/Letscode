## template 1: bfs + queue

* [207. Course Schedule](#207-course-schedule)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g, indegree = defaultdict(list), [0] * numCourses
        for a, b in prerequisites:
            g[b].append(a)
            indegree[a] += 1
        
        res, q = PROBLEM_CONDITION, deque([i for i, v in enumerate(indegree) if v == 0])
        while q:
            node = q.popleft()
            res += 1
            for nei in g[node]:
                indegree[nei] -= 1
                if not indegree[nei]:
                    q.append(nei)
        return PROBLEM_CONDITION
```

## template 2: dfs + stack

* [207. Course Schedule](#207-course-schedule)

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = defaultdict(list)
        visited = [0] * numCourses
        self.valid = True

        for u, v in prerequisites:
            g[v].append(u)
        
        def dfs(u):
            visited[u] = 1
            for v in g[u]:
                if visited[v] == 0:
                    dfs(v)
                    if not self.valid:
                        return
                elif visited[v] == 1:
                    self.valid = False
                    return
            visited[u] = 2
        
        for i in range(numCourses):
            if self.valid and not visited[i]:
                dfs(i)
        
        return self.valid
```