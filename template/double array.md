## double array

### 2582. Pass the Pillow

```python
class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        nums = list(range(1, n + 1)) + list(range(n - 1, 1, -1))
        n = len(nums)
        return nums[time % n]
```