## basic(9)

* [34. Find First and Last Position of Element in Sorted Array](#34-find-first-and-last-position-of-element-in-sorted-array)
* [35. Search Insert Position](#35-search-insert-position)
* [704. Binary Search](#704-binary-search)
* [744. Find Smallest Letter Greater Than Target](#744-find-smallest-letter-greater-than-target)
* [2529. Maximum Count of Positive Integer and Negative Integer](#2529-maximum-count-of-positive-integer-and-negative-integer)
* [1385. Find the Distance Value Between Two Arrays](#1385-find-the-distance-value-between-two-arrays)
* [2300. Successful Pairs of Spells and Potions](#2300-successful-pairs-of-spells-and-potions)
* [2389. Longest Subsequence With Limited Sum](#2389-longest-subsequence-with-limited-sum)
* [2080. Range Frequency Queries](#2080-range-frequency-queries)
* [2563. Count the Number of Fair Pairs](#2563-count-the-number-of-fair-pairs)
* [2856. Minimum Array Length After Pair Removals]()
* [243. Shortest Word Distance](#243-shortest-word-distance)
* [244. Shortest Word Distance II](#244-shortest-word-distance-ii)
* [245. Shortest Word Distance III](#245-shortest-word-distance-iii)
* [374. Guess Number Higher or Lower](#374-guess-number-higher-or-lower)

### 34. Find First and Last Position of Element in Sorted Array

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        def lower_bound(nums, target):
            l, r = 0, n - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] < target:
                    l = m + 1
                else:
                    r = m - 1
            return l
        start = lower_bound(nums, target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        end = lower_bound(nums, target + 1) - 1
        return [start, end]
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        start = bisect_left(nums, target)
        if start == n or nums[start] != target:
            return [-1, -1]
        return [bisect_left(nums, target), bisect_right(nums, target) - 1]
```

### 35. Search Insert Position

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        start = bisect_left(nums, target)
        if start == n or nums[start] != target:
            return [-1, -1]
        end = bisect_left(nums, target + 1) - 1
        return [start, end]
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect_left(nums, target)
```

### 704. Binary Search

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        i = bisect_left(nums, target)
        if i < len(nums) and nums[i] == target:
            return i 
        return -1
```

### 744. Find Smallest Letter Greater Than Target

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        n = len(letters)
        i = bisect_left(letters, chr(ord(target) + 1))
        return letters[i] if i < n else letters[0]
```

### 2529. Maximum Count of Positive Integer and Negative Integer

```python
class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        n = len(nums)
        i = bisect_left(nums, 0)
        j = bisect_right(nums, 0)
        pos, neg = n - j, n - (n - j) - (j - i) 
        return max(pos, neg)
```

### 1385. Find the Distance Value Between Two Arrays

```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        arr2.sort()
        arr2 = [-inf] + arr2 + [inf]
        res = 0
        for n in arr1:
            i = bisect_left(arr2, n)
            if abs(arr2[i] - n) > d and abs(arr2[i - 1] - n) > d:
                res += 1
        return res
```

### 2300. Successful Pairs of Spells and Potions

```python
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        n = len(spells)
        res = [0] * n
        potions.sort()
        for i, v in enumerate(spells):
            j = bisect_left(potions, ceil(success / v))
            res[i] = len(potions) - j 
        return res 
```

### 2389. Longest Subsequence With Limited Sum

```python
class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        a = list(accumulate(nums, initial = 0))
        res = [0] * len(queries)
        for i, q in enumerate(queries):
            j = bisect_left(a, q)
            if j < len(a) and a[j] == q:
                res[i] = j 
            else:
                res[i] = j - 1
        return res
```

### 1170. Compare Strings by Frequency of the Smallest Character

```python
class Solution:
    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        def f(w):
            d = Counter(w)
            for c in ascii_lowercase:
                if c in d:
                    return d[c]
        res = []
        count = [f(w) for w in words]
        for q in queries:
            ans = f(q)
            cnt = 0
            for c in count:
                if ans < c:
                    cnt += 1
            res.append(cnt)
        return res
```

### 2080. Range Frequency Queries

```python
class RangeFreqQuery:

    def __init__(self, arr: List[int]):
        self.d = defaultdict(list)
        for i, n in enumerate(arr):
            self.d[n].append(i)

    def query(self, left: int, right: int, value: int) -> int:
        a = self.d[value]
        l = bisect_left(a, left)
        r = bisect_right(a, right)
        return r - l
```

### 2563. Count the Number of Fair Pairs

```python
class Solution:
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        nums.sort()
        res = 0
        for i, v in enumerate(nums):
            l = bisect_left(nums, lower - v, 0, i)
            r = bisect_right(nums, upper - v, 0, i)
            res += r - l 
        return res 
```

### 2856. Minimum Array Length After Pair Removals

```python
# nums = [1,1,2,2,2]
        d = Counter(nums)
        maxNum = max(d.values())
        n = len(nums)
        if n - (n - maxNum) * 2 >= 0:
            return n - (n - maxNum) * 2
        else:
            if n % 2 == 0:
                return 0
            return 1
```

### 1146. Snapshot Array

```python
class SnapshotArray:

    def __init__(self, length: int):
        self.snaps = [{0: 0} for i in range(length)]
        self.id = 0

    def set(self, index: int, val: int) -> None:
        self.snaps[index][self.id] = val

    def snap(self) -> int:
        self.id += 1
        return self.id - 1

    def get(self, index: int, snap_id: int) -> int:
        if snap_id in self.snaps[index]:
            return self.snaps[index][snap_id]
        arr = list(self.snaps[index].keys())
        i = bisect_left(arr, snap_id)
        return self.snaps[index][arr[i - 1]]
```

### 1818. Minimum Absolute Sum Difference

```python
class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        res, mx, mod = 0, 0, 10 ** 9 + 7
        nums = [-inf] + nums1 + [inf]
        nums.sort()
        for i, (a, b) in enumerate(zip(nums1, nums2)):
            # [-inf, 1, 5, 7, inf]  1
            res += abs(b - a)
            j = bisect_left(nums, b)
            mn = min(b - nums[j - 1], nums[j] - b)
            mx = max(mx, abs(b - a) - mn)
        return (res - mx) % mod
```
### 981. Time Based Key-Value Store

### 911. Online Election

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

### 374. Guess Number Higher or Lower

```python
class Solution:
    def guessNumber(self, n: int) -> int:
        l, r = 1, n
        while l <= r:
            m = l + (r - l) // 2
            if guess(m) == 1:
                l = m + 1
            elif guess(m) == -1:
                r = m - 1
            else:
                return m
```