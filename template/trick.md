## tricks

* [2960. Count Tested Devices After Test Operations](#2960-count-tested-devices-after-test-operations)

### 2960. Count Tested Devices After Test Operations

- reverse thinking: just use one value to compare with last values

```python
class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        n = 0
        for b in batteryPercentages:
            if b > n:
                n += 1
        return n 
```

### 2017. Grid Game

```python
class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        pre = []
        for row in grid:
            pre.append(list(accumulate(row, initial = 0)))
        res = inf
        for i in range(1, len(pre[0])):
            res = min(res, max(pre[0][-1] - pre[0][i], pre[1][i - 1]))
        return res
```