## simulation

- brute force

### Simulation questions

* [1402. Reducing Dishes](#1402-Reducing-Dishes)
* [2437. Number of Valid Clock Times](#2437-Number-of-Valid-Clock-Times)
* [2456. Most Popular Video Creator](#2456-Most-Popular-Video-Creator)
* [2409. Count Days Spent Together](#2409-Count-Days-Spent-Together)
* [2512. Reward Top K Students](#2512-Reward-Top-K-Students)
* [1402. Reducing Dishes](#)
* [1402. Reducing Dishes](#)
* [1402. Reducing Dishes](#)


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