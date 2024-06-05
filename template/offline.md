### 2343. Query Kth Smallest Trimmed Number

```python
class Solution:
    def smallestTrimmedNumbers(self, nums: List[str], queries: List[List[int]]) -> List[int]:
        index = list(range(len(nums)))
        res, i = [0] * len(queries), 1
        for q_index, (k, trim) in sorted(enumerate(queries), key = lambda x: x[1][1]):
            while i <= trim:
                index.sort(key = lambda x: nums[x][-i])
                i += 1
            res[q_index] = index[k - 1]
        return res
```

### 2070. Most Beautiful Item for Each Query

```python
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        index = list(range(len(queries)))
        res = [0] * len(queries)
        items.sort()
        index.sort(key = lambda x: queries[x])
        i, max_beauty = 0, 0
        for j, q in enumerate(sorted(queries)):
            while i < len(items) and items[i][0] <= q:
                max_beauty = max(max_beauty, items[i][1])
                i += 1
            res[index[j]] = max_beauty
        return res
```

### 1847. Closest Room

```python
from sortedcontainers import SortedList
class Solution:
    def closestRoom(self, rooms: List[List[int]], queries: List[List[int]]) -> List[int]:
        index = list(range(len(queries)))
        index.sort(key = lambda x: queries[x][1], reverse = True)
        rooms.sort(key = lambda x: x[1], reverse = True)
        res = [-1] * len(queries)
        i = 0
        sl = SortedList([-inf, inf])
        for j, (preferred, mn_size) in enumerate(sorted(queries, key = lambda x: -x[1])):   
            while i < len(rooms) and rooms[i][1] >= mn_size:
                sl.add(rooms[i][0]) # add room id 
                i += 1
            pos = sl.bisect_left(preferred)
            if preferred - sl[pos - 1] <= sl[pos] - preferred:
                room_id = sl[pos - 1]
            else:
                room_id = sl[pos]
            res[index[j]] = room_id if room_id != -inf else -1
        return res
```

### 2503. Maximum Number of Points From Grid Queries

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
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        R, C, directions = len(grid), len(grid[0]), [[0, 1], [0, -1], [1, 0], [-1, 0]]
        nums = sorted((grid[r][c], r, c) for r in range(R) for c in range(C))
        index = list(range(len(queries)))
        index.sort(key = lambda x: queries[x])
        res = [0] * len(queries)
        i = 0
        n = R * C
        uf = UF(n)
        for j, q in enumerate(sorted(queries)):
            while i < n and nums[i][0] < q:
                val, r, c = nums[i]
                for dr, dc in directions:
                    row, col = r + dr, c + dc 
                    if 0 <= row < R and 0 <= col < C and grid[row][col] < q and not uf.isConnected(r * C + c, row * C + col):
                        uf.union(r * C + c, row * C + col)
                i += 1
            if grid[0][0] < q:
                res[index[j]] = uf.rank[uf.find(0)]
        return res
```