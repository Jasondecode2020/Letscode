### 1310. XOR Queries of a Subarray

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        arr = [0] + arr
        for i in range(1, len(arr)):
            arr[i] ^= arr[i - 1]
        res = []
        for s, e in queries:
            res.append(arr[e + 1] ^ arr[s])
        return res
```