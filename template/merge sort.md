## merge sort

### 786. K-th Smallest Prime Fraction

```python
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        pq = []
        for i in range(1, len(arr)): # res, denominator, numerator
            heapq.heappush(pq, (1/arr[i], i, 0))
        for _ in range(k):
            val, j, i = heapq.heappop(pq)
            if i < j - 1:
                heapq.heappush(pq, (arr[i+1] / arr[j], j, i+1))
        return [arr[i], arr[j]]
```