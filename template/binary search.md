## template 1: standard search en element

```python
def fn(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if arr[mid] == target:
            # some code or return target if find
            return
        if arr[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    
    # l is insertion point
    return l
```

## template 2: duplicate element insert at left

```python
def fn(arr, target):
    l, r = 0, len(arr)
    while l < r:
        mid = l + (r - l) // 2
        if arr[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l
```

## template 3: duplicate element insert at the right

```python
def fn(arr, target):
    l, r = 0, len(arr)
    while l < r:
        mid = l + (r - l) // 2
        if arr[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l
```

### 243. Shortest Word Distance

```python
class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        a, b = [], []
        for i, w in enumerate(wordsDict):
            if w == word1:
                a.append(i) 
            if w == word2:
                b.append(i)
        a.sort()
        b.sort()
        res = inf
        b = [-inf] + b + [inf]
        for n in a:
            idx = bisect_left(b, n)
            res = min(res, b[idx] - n, n - b[idx - 1])
        return res
```

### 244. Shortest Word Distance II

```python
class WordDistance:

    def __init__(self, wordsDict: List[str]):
        self.words = defaultdict(list)
        for i, w in enumerate(wordsDict):
            self.words[w].append(i)

    def shortest(self, word1: str, word2: str) -> int:
        a = self.words[word1]
        b = self.words[word2]
        res = inf
        b = [-inf] + b + [inf]
        for n in a:
            idx = bisect_left(b, n)
            res = min(res, b[idx] - n, n - b[idx - 1])
        return res
```

### 245. Shortest Word Distance III

```python
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        words = defaultdict(list)
        for i, w in enumerate(wordsDict):
            words[w].append(i)
        
        res = inf
        if word1 != word2:
            a = words[word1]
            b = words[word2]
            b = [-inf] + b + [inf]
            for n in a:
                idx = bisect_left(b, n)
                res = min(res, b[idx] - n, n - b[idx - 1])
        else:
            a = words[word1]
            res = min(a[i] - a[i - 1] for i in range(1, len(a)))
        return res
```

