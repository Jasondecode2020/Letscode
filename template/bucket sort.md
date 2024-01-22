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