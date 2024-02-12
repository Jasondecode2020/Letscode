### bucket sort

### 6. Zigzag Conversion

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        bucket, flip = [[] for i in range(numRows)], -1
        i = 0
        for c in s:
            bucket[i].append(c)
            if i == 0 or i == numRows - 1:
                flip *= -1
            i += flip
        for i, b in enumerate(bucket):
            bucket[i] = ''.join(b)
        return ''.join(bucket)
```

### 164. Maximum Gap

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        L = len(nums)
        if L < 2:
            return 0

        mn, mx = min(nums), max(nums)
        if mn == mx:
            return 0
        d = ceil((mx - mn) / (L - 1))
        buckets = [[] for _ in range(L + 1)]
        for n in nums:
            buckets[(n - mn) // d].append(n)
        res = 0
        bucket_max = max(buckets[0])
        for i in range(1, L + 1):
            if not buckets[i]:
                continue
            bucket_min = min(buckets[i])
            res = max(res, bucket_min - bucket_max)
            bucket_max = max(buckets[i])
        return res
```