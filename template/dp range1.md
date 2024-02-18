### 3041. Maximize Consecutive Elements in an Array After Modification

```python
class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        d = defaultdict(int)
        nums.sort()
        for x in nums:
            d[x + 1] = d[x] + 1
            d[x] = d[x - 1] + 1
        return max(d.values())
```