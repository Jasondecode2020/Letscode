### 492. Construct the Rectangle

```python
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        ans = inf 
        res = []
        for i in range(1, int(sqrt(area)) + 1):
            if area % i == 0 and abs(i - area // i) < ans:
                res = [area // i, i]
        return res
```