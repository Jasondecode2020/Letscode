### 2001. Number of Pairs of Interchangeable Rectangles

```python
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        ratio = Counter()
        for a, b in rectangles:
            # ratio[a / b] += 1
            ratio[(a // gcd(a, b), b // gcd(a, b))] += 1
        res = 0
        for v in ratio.values():
            res += v * (v - 1) // 2
        return res
```

### 2280. Minimum Lines to Represent a Line Chart

```python
class Solution:
    def minimumLines(self, stockPrices: List[List[int]]) -> int:
        def check(x, y):
            x1, y1 = x
            x2, y2 = y
            dy, dx = (y2 - y1) // gcd(y2 - y1, x2 - x1), (x2 - x1) // gcd(y2 - y1, x2 - x1)
            return (dy, dx)

        stockPrices.sort()
        n = len(stockPrices)
        pre = (inf, inf)
        res = 0
        for i in range(1, n):
            slope = check(stockPrices[i - 1], stockPrices[i])
            if slope != pre:
                res += 1
                pre = slope 
        return res
```

### 2436. Minimum Split Into Subarrays With GCD Greater Than One

```python
class Solution:
    def minimumSplits(self, nums: List[int]) -> int:
        nums.append(10 ** 9 + 7)
        n = len(nums)
        res = 0
        for i in range(1, n):
            if gcd(nums[i - 1], nums[i]) > 1:
                nums[i] = gcd(nums[i - 1], nums[i])
            else:
                res += 1
        return res
```