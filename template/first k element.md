## template

```python
import heapq
def fn(arr, k):
    heap = []
    for num in arr:
        # some code
        heapq.heappush(heap, (CRITERIA, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [num for num in heap]
```

## top k element

### 973. K Closest Points to Origin

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points = [(x ** 2 + y ** 2, x, y) for x, y in points]
        points.sort()
        res = []
        for i in range(k):
            res.append([points[i][1], points[i][2]])
        return res
```

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key = lambda x: x[0] ** 2 + x[1] ** 2)
        return points[:k]
```