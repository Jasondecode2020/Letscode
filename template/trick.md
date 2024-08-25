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

### 288. Unique Word Abbreviation

```python
class ValidWordAbbr:

    def __init__(self, dictionary: List[str]):
        self.d = defaultdict(int)
        self.word = set(dictionary)
        for word in self.word:
            if len(word) <= 2:
                self.d[word] += 1
            else:
                self.d[word[0] + str((len(word) - 2)) + word[-1]] += 1

    def isUnique(self, word: str) -> bool:
        origin = word
        word = word if len(word) <= 2 else word[0] + str((len(word) - 2)) + word[-1]
        if origin in self.word and self.d[word] == 1:
            return True
        return not word in self.d
```