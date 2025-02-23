## basic(9)

* [34. Find First and Last Position of Element in Sorted Array](#34-find-first-and-last-position-of-element-in-sorted-array)
* [35. Search Insert Position](#35-search-insert-position)
* [704. Binary Search](#704-binary-search)
* [744. Find Smallest Letter Greater Than Target](#744-find-smallest-letter-greater-than-target)
* [2529. Maximum Count of Positive Integer and Negative Integer](#2529-maximum-count-of-positive-integer-and-negative-integer)

* [1385. Find the Distance Value Between Two Arrays](#1385-find-the-distance-value-between-two-arrays)
* [2300. Successful Pairs of Spells and Potions](#2300-successful-pairs-of-spells-and-potions)
* [2389. Longest Subsequence With Limited Sum](#2389-longest-subsequence-with-limited-sum)
* [1170. Compare Strings by Frequency of the Smallest Character](#1170-compare-strings-by-frequency-of-the-smallest-character)
* [2080. Range Frequency Queries](#2080-range-frequency-queries)

* [2563. Count the Number of Fair Pairs](#2563-count-the-number-of-fair-pairs)
* [981. Time Based Key-Value Store](#981-time-based-key-value-store)
* [1146. Snapshot Array](#1146-snapshot-array)
* [1818. Minimum Absolute Sum Difference](#1818-minimum-absolute-sum-difference)
* [911. Online Election](#911-online-election)

* [658. Find K Closest Elements](#658-find-k-closest-elements)
* [1064. Fixed Point](#1064-fixed-point)
* [1150. Check If a Number Is Majority Element in a Sorted Array](#1150-check-if-a-number-is-majority-element-in-a-sorted-array)
* [1182. Shortest Distance to Target Color](#1182-shortest-distance-to-target-color)
* [702. Search in a Sorted Array of Unknown Size]()

* [2856. Minimum Array Length After Pair Removals](#2856-minimum-array-length-after-pair-removals)
* [243. Shortest Word Distance](#243-shortest-word-distance)
* [244. Shortest Word Distance II](#244-shortest-word-distance-ii)
* [245. Shortest Word Distance III](#245-shortest-word-distance-iii)
* [374. Guess Number Higher or Lower](#374-guess-number-higher-or-lower)

* [1095. Find in Mountain Array](#1095-find-in-mountain-array)

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
        l = bisect_left(nums, target)
        if l == len(nums) or nums[l] != target:
            return [-1, -1]
        r = bisect_right(nums, target) - 1
        return [l, r]
```

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

### 35. Search Insert Position

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return l 
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
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                return m 
        return -1
```

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

class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        i = bisect_right(letters, target)
        if i == len(letters):
            return letters[0]
        return letters[i]
```

### 2529. Maximum Count of Positive Integer and Negative Integer

```python
class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        n = len(nums)
        i = bisect_left(nums, 0)
        j = bisect_right(nums, 0)
        pos = n - j
        neg = i
        return max(pos, neg)
```

### 1385. Find the Distance Value Between Two Arrays

```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        arr2 = [-inf] + arr2 + [inf]
        arr2.sort()
        res = 0
        for n in arr1:
            i = bisect_left(arr2, n)
            if n - arr2[i - 1] > d and arr2[i] - n > d:
                res += 1
        return res 
```

### 2300. Successful Pairs of Spells and Potions

```python
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        n = len(spells)
        pairs = []
        potions.sort()
        for n in spells:
            i = bisect_left(potions, ceil(success / n))
            pairs.append(len(potions) - i)
        return pairs
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

class Solution:
    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        def f(word):
            d = Counter(word)
            return d[sorted(d.keys())[0]]

        count = [f(w) for w in words]
        count.sort()
        res = []
        for q in queries:
            q = f(q)
            i = bisect_right(count, q)
            res.append(len(count) - i)
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
        arr = self.d[value]
        l = bisect_left(arr, left)
        r = bisect_right(arr, right)
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

### 981. Time Based Key-Value Store

```python 
class TimeMap:

    def __init__(self):
        self.d = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.d[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        i = bisect_right(self.d[key], timestamp, key = lambda x: x[0])
        if i != 0:
            return self.d[key][i - 1][-1]
        return ''
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
        return self.snaps[index][arr[i - 1]] if i > 0 else 0
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

### 911. Online Election

```python 
class TopVotedCandidate:

    def __init__(self, persons: List[int], times: List[int]):
        n = len(times)
        self.res, self.times = [-1] * n, times 
        self.d, cur = defaultdict(int), None
        for i in range(n):
            self.d[persons[i]] += 1
            if cur is None or self.d[persons[i]] >= self.d[cur]:
                cur = persons[i]
            self.res[i] = cur 
        
    def q(self, t: int) -> int:
        return self.res[bisect_right(self.times, t) - 1]
```

### 658. Find K Closest Elements

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l, left, right, total = 0, -1, -1, 0
        ans = inf
        for r, n in enumerate(arr):
            total += abs(n - x)
            if r - l + 1 == k:
                if total < ans:
                    ans = total
                    left, right = l, r
                total -= abs(arr[l] - x)
                l += 1
        return arr[left: right + 1]
```

### 1064. Fixed Point

```python 
class Solution:
    def fixedPoint(self, arr: List[int]) -> int:
        l, r = 0, len(arr) - 1
        res = inf 
        while l <= r:
            m = (l + r) // 2
            if arr[m] == m:
                res = min(res, m)
            if arr[m] >= m:
                r = m - 1
            else:
                l = m + 1
        return res if res != inf else -1
```

### 1150. Check If a Number Is Majority Element in a Sorted Array

```python 
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        l = bisect_left(nums, target)
        r = bisect_right(nums, target)
        return r - l > len(nums) // 2
```


### 1182. Shortest Distance to Target Color

```python
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        d = defaultdict(list)
        for i, color in enumerate(colors):
            d[color].append(i)
        res = []
        for i, c in queries:
            j = bisect_left(d[c], i)
            if len(d[c]) == 0:
                res.append(-1)
            else:
                if j < len(d[c]):
                    res.append(min(abs(i - d[c][j - 1]), abs(d[c][j] - i)))
                else:
                    res.append(i - d[c][-1])
        return res 
```

### 702. Search in a Sorted Array of Unknown Size

```python 
class Solution:
    def search(self, reader: 'ArrayReader', target: int) -> int:
        l, r = 0, 10 ** 4 - 1
        while l <= r:
            m = (l + r) // 2
            if reader.get(m) < target:
                l = m + 1
            elif reader.get(m) > target:
                r = m - 1
            else:
                return m 
        return -1
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

### 1095. Find in Mountain Array

```python
class Solution:
    def findInMountainArray(self, target: int, mountainArr: 'MountainArray') -> int:
        def findPeak():
            l, r = 1, mountainArr.length() - 2
            res = 0
            while l <= r:
                m = (l + r) // 2
                if mountainArr.get(m) > mountainArr.get(m + 1):
                    res = m 
                    r = m - 1
                else:
                    l = m + 1
            return res 
        pIndex = findPeak()
        def binary_search(l, r):
            while l <= r:
                m = (l + r) // 2
                if mountainArr.get(m) > target:
                    r = m - 1
                elif mountainArr.get(m) < target:
                    l = m + 1
                else:
                    return m 
            return -1
        def binary_search2(l, r):
            while l <= r:
                m = (l + r) // 2
                if mountainArr.get(m) > target:
                    l = m + 1
                elif mountainArr.get(m) < target:
                    r = m - 1
                else:
                    return m 
            return -1
        idx = -1
        idx = binary_search(0, pIndex)
        if idx != -1:
            return idx
        return binary_search2(pIndex, mountainArr.length() - 1)
```