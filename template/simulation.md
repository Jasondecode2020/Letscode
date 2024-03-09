## simulation

- brute force

### Simulation questions

* [1402. Reducing Dishes](#1402-Reducing-Dishes)
* [2437. Number of Valid Clock Times](#2437-Number-of-Valid-Clock-Times)
* [2456. Most Popular Video Creator](#2456-Most-Popular-Video-Creator)
* [2409. Count Days Spent Together](#2409-Count-Days-Spent-Together)
* [2512. Reward Top K Students](#2512-Reward-Top-K-Students)
* [2502. Design Memory Allocator](#2502-Design-Memory-Allocator)
* [2353. Design a Food Rating System](#2353-Design-a-Food-Rating-System)


### 1402. Reducing Dishes

```python
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        def check(nums):
            ans = 0
            for i, v in enumerate(nums):
                ans += (i + 1) * v
            return ans
        res = 0
        satisfaction.sort()
        while satisfaction:
            res = max(res, check(satisfaction))
            satisfaction.pop(0)
        return res
```

### 2437. Number of Valid Clock Times

- hour and minute are independant

```python
class Solution:
    def countTime(self, time: str) -> int:
        def checkMinute(m1, m2):
            if m1 + m2 == '??':
                return 60
            elif m1 != '?' and m2 == '?':
                return 10
            elif m1 == '?' and m2 != '?':
                return 6
            return 1

        def checkHour(h1, h2):
            if h1 + h2 == '??':
                return 24
            elif h1 == '?' and h2 != '?':
                if int(h2) <= 3:
                    return 3
                else:
                    return 2
            elif h1 != '?' and h2 == '?':
                if int(h1) <= 1:
                    return 10
                else:
                    return 4
            elif h1 != '?' and h2 != '?':
                return 1

        h1, h2, _, m1, m2 = time
        return checkHour(h1, h2) * checkMinute(m1, m2)
```

### 2349. Design a Number Container System

- defaultdict(SortedList)

```python
from sortedcontainers import SortedList
class NumberContainers:

    def __init__(self):
        self.d = defaultdict(int)
        self.d_sl = defaultdict(SortedList)

    def change(self, index: int, number: int) -> None:
        if index in self.d:
            oldNum = self.d[index]
            self.d_sl[oldNum].remove(index)

        self.d[index] = number
        self.d_sl[number].add(index)

    def find(self, number: int) -> int:
        return self.d_sl[number][0] if self.d_sl[number] else -1
```

### 2456. Most Popular Video Creator

```python
class Solution:
    def mostPopularCreator(self, creators: List[str], ids: List[str], views: List[int]) -> List[List[str]]:
        d_view = defaultdict(int)
        d_pop = defaultdict(list)
        for c, i, v in zip(creators, ids, views):
            d_view[c] += v
            d_pop[c].append((v, i))

        maxView = max(d_view.values())
        res = []
        for c in d_view:
            if d_view[c] == maxView:
                res.append([c, sorted(d_pop[c], key = lambda x: (-x[0], x[1]))[0][1]])
        return res
```

### 2409. Count Days Spent Together

```python
class Solution:
    def countDaysTogether(self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str) -> int:
        month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        def checkDay(m, d):
            res = 0
            m, d = int(m), int(d)
            for i in range(m - 1):
                res += month[i]
            return res + d

        startA = checkDay(arriveAlice[:2], arriveAlice[3:])
        endA = checkDay(leaveAlice[:2], leaveAlice[3:])
        startB = checkDay(arriveBob[:2], arriveBob[3:])
        endB = checkDay(leaveBob[:2], leaveBob[3:])
        if max(startA, startB) <= min(endA, endB):
            return min(endA, endB) - max(startA, startB) + 1
        return 0
```

### 2512. Reward Top K Students

```python
class Solution:
    def topStudents(self, positive_feedback: List[str], negative_feedback: List[str], report: List[str], student_id: List[int], k: int) -> List[int]:
        positive_feedback = set(positive_feedback)
        negative_feedback = set(negative_feedback)
        d = defaultdict(int)
        for s, r in zip(student_id, report):
            for w in r.split(' '):
                if w in positive_feedback:
                    d[s] += 3
                elif w in negative_feedback:
                    d[s] -= 1
                else:
                    d[s] += 0
        res = []
        for i, v in d.items():
            res.append((v, i))
        res.sort(key = lambda x: (-x[0], x[1]))
        return [i for v, i in res][: k]
```

### 2502. Design Memory Allocator

```python
class Allocator:

    def __init__(self, n: int):
        self.res = [0] * n

    def allocate(self, size: int, mID: int) -> int:
        count = 0
        for i, id in enumerate(self.res):
            if id:
                count = 0
            else:
                count += 1
                if count == size:
                    self.res[i - size + 1: i + 1] = [mID] * size
                    return i - size + 1
        return -1

    def free(self, mID: int) -> int:
        res = 0
        for i in range(len(self.res)):
            if self.res[i] == mID:
                self.res[i] = 0
                res += 1
        return res
```

### 2353. Design a Food Rating System

```python
from sortedcontainers import SortedList
class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.d = defaultdict(SortedList)
        self.food2cuisine = {}
        self.food2rating = {}
        for f, c, r in zip(foods, cuisines, ratings):
            self.d[c].add((-r, f))
            self.food2cuisine[f] = c
            self.food2rating[f] = -r

    def changeRating(self, food: str, newRating: int) -> None:
        c = self.food2cuisine[food]
        r = self.food2rating[food]
        self.d[c].remove((r, food))
        self.d[c].add((-newRating, food))
        self.food2rating[food] = -newRating

    def highestRated(self, cuisine: str) -> str:
        return self.d[cuisine][0][1]
```

### 1243. Array Transformation

```python
class Solution:
    def transformArray(self, arr: List[int]) -> List[int]:
        flag, n, res = True, len(arr), []
        while flag:
            flag = False
            for i in range(n):
                if i == 0 or i == n - 1:
                    res.append(arr[i])
                elif arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                    res.append(arr[i] - 1)
                    flag = True
                elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                    res.append(arr[i] + 1)
                    flag = True
                else:
                    res.append(arr[i])
            arr = res
            res = []
        return arr
```

### 243. Shortest Word Distance

```python
class Solution:
    def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        res, a, b = inf, -inf, inf
        for i, w in enumerate(wordsDict):
            if w == word1:
                a = i 
            elif w == word2:
                b = i 
            res = min(res, abs(a - b))
        return res
```

### 2274. Maximum Consecutive Floors Without Special Floors

```python
class Solution:
    def maxConsecutive(self, bottom: int, top: int, special: List[int]) -> int:
        special.sort()
        diff = [special[i] - special[i - 1] - 1 for i in range(1, len(special))]
        return max(*diff, special[0] - bottom, top - special[-1])
```

### 2161. Partition Array According to Given Pivot

```python
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        res1, res2, res3 = [], [], []
        for n in nums:
            if n < pivot:
                res1.append(n)
            elif n > pivot:
                res3.append(n)
            else:
                res2.append(n)
        return res1 + res2 + res3
```

### 1138. Alphabet Board Path

```python
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
        d = {}
        for i, b in enumerate(board):
            for j, c in enumerate(b):
                d[c] = (j, i)
        
        x1, y1 = (0, 0)
        res = ''
        for c in target:
            x2, y2 = d[c]
            dx, dy = x2 - x1, y2 - y1
            if dx <= 0:
                res += 'L' * abs(dx)
            if dy > 0:
                res += 'D' * dy 
            else:
                res += 'U' * abs(dy)
            if dx > 0:
                res += 'R' * dx
            res += '!'
            x1, y1 = x2, y2
        return res
```