## sorting: to sort arr simulation ans

- 768. Max Chunks To Make Sorted II
- 769. Max Chunks To Make Sorted


### 768. Max Chunks To Make Sorted II

```python
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        c, res = defaultdict(int), 0
        for a, b in zip(arr, sorted(arr)):
            c[a] += 1
            if c[a] == 0:
                del c[a]
            c[b] -= 1
            if c[b] == 0:
                del c[b]
            if len(c) == 0:
                res += 1
        return res
```

### 769. Max Chunks To Make Sorted

```python
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        c, res = defaultdict(int), 0
        for a, b in zip(arr, sorted(arr)):
            c[a] += 1
            if c[a] == 0:
                del c[a]
            c[b] -= 1
            if c[b] == 0:
                del c[b]
            if len(c) == 0:
                res += 1
        return res
```

### 912. Sort an Array

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # [5,2,3,1]
        if len(nums) <= 1:
            return nums
        
        index = random.randint(0, len(nums) - 1)
        pivot = nums[index]

        greater = [n for n in nums if n > pivot]
        equal = [n for n in nums if n == pivot]
        less = [n for n in nums if n < pivot]
        return self.sortArray(less) + equal + self.sortArray(greater)
```

## bucket sort

### 347. Top K Frequent Elements

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # method 1: dict
        # c = Counter(nums)
        # print(c.most_common(2))
        # return [item[0] for item in c.most_common(k)]

        # method 2: heap
        # c = Counter(nums)
        # pq = [(v, k) for k, v in c.items()]
        # return [item[1] for item in nlargest(k, pq)]

        # method 3: bucket sort
        c, n, res = Counter(nums), len(nums), []
        bucket = [[] for _ in range(n + 1)]
        for x, v in c.items():
            bucket[v].append(x)

        for v in range(n, -1, -1):
            res.extend(bucket[v])
        return res[: k]
```

### 912. Sort an Array

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # [5,2,3,1]
        if len(nums) <= 1:
            return nums
        
        index = random.randint(0, len(nums) - 1)
        pivot = nums[index]

        greater = [n for n in nums if n > pivot]
        equal = [n for n in nums if n == pivot]
        less = [n for n in nums if n < pivot]
        return self.sortArray(less) + equal + self.sortArray(greater)
```

### 436. Find Right Interval

```python
class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        res = []
        for i, (s, e) in enumerate(intervals):
            res.append([s, e, i])
        res.sort()
        n = len(intervals)
        ans = [-1] * n
        for s, e, i in res:
            idx = bisect_left(res, e, key = lambda x: x[0])
            if idx < n:
                j = res[idx][2]
                ans[i] = j 
        return ans
```

### 475. Heaters

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        ans = -inf
        heaters = [-inf] + heaters + [inf]
        heaters.sort()
        for h in houses:
            i = bisect_left(heaters, h)
            res = min(h - heaters[i - 1], heaters[i] - h)
            ans = max(res, ans)
        return ans
```

### 791. Custom Sort String

```python
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        d = defaultdict(int)
        for i, c in enumerate(order):
            d[c] = i 
        res = []
        for c in s:
            if c in d:
                res.append((d[c], c))
            else:
                res.append((26, c))
        res.sort()
        return ''.join([item[1] for item in res])
```

### 846. Hand of Straights

```python
from sortedcontainers import SortedList
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        n = len(hand)
        if n % groupSize:
            return False

        sl = SortedList(hand)
        while sl:
            mn = sl[0]
            for i in range(groupSize):
                if mn in sl:
                    sl.remove(mn)
                    mn += 1
                else:
                    return False
        return True
```

### 1296. Divide Array in Sets of K Consecutive Numbers

```python
from sortedcontainers import SortedList
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n % k:
            return False

        sl = SortedList(nums)
        while sl:
            mn = sl[0]
            for i in range(k):
                if mn in sl:
                    sl.remove(mn)
                    mn += 1
                else:
                    return False
        return True
```

### 1366. Rank Teams by Votes

```python
class Solution:
    def rankTeams(self, votes: List[str]) -> str:
        n = len(votes[0])
        ranking = collections.defaultdict(lambda: [0] * n)
        for vote in votes:
            for i, c in enumerate(vote):
                ranking[c][i] += 1
        result = list(ranking.items())
        result.sort(key=lambda x: (x[1], -ord(x[0])), reverse=True)
        return "".join([c for c, rank in result])
```