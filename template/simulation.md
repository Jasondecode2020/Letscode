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
* [755. Pour Water](#755-pour-water)
* [927. Three Equal Parts](#927-three-equal-parts)

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

### 1409. Queries on a Permutation With Key

```python
class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        a = list(range(1, m + 1))
        res = []
        for q in queries:
            i = a.index(q)
            res.append(i)
            a = [a[i]] + a[: i] + a[i + 1:]
        return res
```

### 1860. Incremental Memory Leak

```python
class Solution:
    def memLeak(self, memory1: int, memory2: int) -> List[int]:
        t = 1
        while memory1 >= 0 and memory2 >= 0:
            if memory1 >= memory2:
                if memory1 - t >= 0:
                    memory1 -= t 
                else:
                    break
            else:
                if memory2 - t >= 0:
                    memory2 -= t
                else:
                    break
            t += 1
        return [t, memory1, memory2]
```

### 2120. Execution of All Suffix Instructions Staying in a Grid

```python
class Solution:
    def executeInstructions(self, n: int, startPos: List[int], s: str) -> List[int]:
        m = len(s)
        res = [0] * m 
        for i in range(m):
            count = 0
            r, c = startPos
            for j in range(i, m):
                if s[j] == 'L' and c - 1 >= 0:
                    c -= 1
                    count += 1
                elif s[j] == 'R' and c + 1 < n:
                    c += 1
                    count += 1
                elif s[j] == 'U' and r - 1 >= 0:
                    r -= 1
                    count += 1
                elif s[j] == 'D' and r + 1 < n:
                    r += 1
                    count += 1
                else:
                    break
            res[i] = count
        return res
```

### 2452. Words Within Two Edits of Dictionary

```python
class Solution:
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        def check(word1, word2):
            res = 0
            for a, b in zip(word1, word2):
                if a != b:
                    res += 1
            return res <= 2 
        res = []
        for q in queries:
            for word in dictionary:
                if check(q, word):
                    res.append(q)
                    break 
        return res
```

### 1535. Find the Winner of an Array Game

```python
class Solution:
    def getWinner(self, arr: List[int], k: int) -> int:
        mx = arr[0]
        count = 0
        for i in range(1, len(arr)):
            if arr[i] > mx:
                count = 1
                mx = arr[i]
            else:
                count += 1
            if count == k:
                return mx
        return mx
```

### 2294. Partition Array Such That Maximum Difference Is K

```python
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        # [1, 2, 3, 5, 6]
        nums = nums + [inf]
        l = 0
        res = 0
        for r, v in enumerate(nums):
            if v - nums[l] > k:
                res += 1
                l = r 
        return res
```

### 2526. Find Consecutive Integers from a Data Stream

```python
class DataStream:

    def __init__(self, value: int, k: int):
        self.value = value
        self.k = k
        self.cnt = 0

    def consec(self, num: int) -> bool:
        self.cnt = 0 if num != self.value else self.cnt + 1
        return self.cnt >= self.k
```

### 2918. Minimum Equal Sum of Two Arrays After Replacing Zeros

```python
class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1, s2 = sum(nums1), sum(nums2)
        zero1, zero2 = nums1.count(0), nums2.count(0)
        p1, p2 = s1 + zero1, s2 + zero2
        if (p1 > s2 and zero2 == 0) or (p2 > s1 and zero1 == 0):
            return -1
        return max(p1, p2)
```

### 2711. Difference of Number of Distinct Values on Diagonals

```python
class Solution:
    def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        res = [[0] * C for r in range(R)]
        for r in range(R):
            for c in range(C):
                topLeft = set()
                x, y = r - 1, c - 1
                while x >= 0 and y >= 0:
                    topLeft.add(grid[x][y])
                    x -= 1
                    y -= 1
                bottomRight = set()
                x, y = r + 1, c + 1
                while x < R and y < C:
                    bottomRight.add(grid[x][y])
                    x += 1
                    y += 1
                res[r][c] = abs(len(topLeft) - len(bottomRight))
        return res
```

### 1387. Sort Integers by The Power Value

```python
class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        def check(n):
            res = 0
            while n != 1:
                if n % 2 == 0:
                    n //= 2
                else:
                    n = n * 3 + 1
                res += 1
            return res
        res = []
        for i in range(lo, hi + 1):
            res.append((check(i), i))
        res.sort()
        return res[k - 1][1]
```

### 2380. Time Needed to Rearrange a Binary String

- O(n^2)

```python
class Solution:
    def secondsToRemoveOccurrences(self, s: str) -> int:
        a = list(s)
        res = 0
        while True:
            flag = False
            i = 0
            while i < len(s) - 1:
                if a[i] == '0' and a[i + 1] == '1':
                    a[i], a[i + 1] = '1', '0'
                    i += 2
                    flag = True
                else:
                    i += 1
            if not flag:
                break
            res += 1
        return res
```

### 950. Reveal Cards In Increasing Order

```python
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        # [17] => [13, 17] => [11, 17, 13] => [7, 13, 11, 17]=> [5, 17, 7, 13, 11]
        # => [3, 11, 5, 17, 7, 13] => [2, 13, 3, 11, 5, 17, 7]
        res = []
        deck.sort(reverse = True)
        for i, card in enumerate(deck):
            if i == 0:
                res.append(card)
            else:
                res = [card] + [res[-1]] + res[:-1]
        return res 
```

### 838. Push Dominoes

```python
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        n = len(dominoes)
        right = [inf] * n 
        cur = -inf 
        for i, d in enumerate(dominoes):
            if d == 'R':
                cur = i 
            elif d == 'L':
                cur = -inf 
            right[i] = i - cur 

        left = [inf] * n 
        cur = inf 
        for i in range(n - 1, -1, -1):
            if dominoes[i] == 'L':
                cur = i 
            elif dominoes[i] == 'R':
                cur = inf 
            left[i] = cur - i 
        return ''.join(['.' if l == r else ('R' if l > r else 'L') for l, r in zip(left, right)])
```

### 755. Pour Water

```python
class Solution:
    def pourWater(self, heights: List[int], volume: int, k: int) -> List[int]:
        n = len(heights)
        for _ in range(volume):
            i = k
            pos = k
            find = False
            for d in [-1, 1]:
                while 0 <= i + d < n and heights[i + d] <= heights[i]:
                    if heights[i + d] < heights[i]:
                        pos = i + d 
                    i += d 
                if pos != k:
                    heights[pos] += 1
                    find = True
                    break
            if not find:
                heights[pos] += 1
        return heights
```

### 927. Three Equal Parts

```python
class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        ones = arr.count(1)
        if ones == 0:
            return [0, len(arr) - 1]
        if ones % 3:
            return [-1, -1] 
        
        target = ones // 3 
        p1 = p2 = p3 = 0
        cnt = 0
        for i, num in enumerate(arr):
            if num == 1:
                if cnt == 0:
                    p1 = i 
                elif cnt == target:
                    p2 = i 
                elif cnt == 2 * target:
                    p3 = i 
                
                cnt += 1 
        
        origin_p2 = p2 
        origin_p3 = p3 
        while p1 < origin_p2 and p2 < origin_p3 and p3 < len(arr):
            if arr[p1] != arr[p2] or arr[p2] != arr[p3]:
                return [-1, -1]
            p1 += 1
            p2 += 1
            p3 += 1
        return [p1 - 1, p2] if p3 == len(arr) else [-1, -1]
```

### 1121. Divide Array Into Increasing Sequences

```python
class Solution:
    def canDivideIntoSubsequences(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        c = Counter(nums)
        mx = max(c.values())
        return n // mx >= k 
```