## heap(30)

* [984. String Without AAA or BBB](#984-string-without-aaa-or-bbb)
* [767. Reorganize String](#767-reorganize-string)
* [1054. Distant Barcodes](#692-top-k-frequent-words)
* [1953. Maximum Number of Weeks for Which You Can Work](#1953-maximum-number-of-weeks-for-which-you-can-work)
* [2974. Minimum Number Game](#2974-minimum-number-game)


### 984. String Without AAA or BBB

```python
class Solution:
    def strWithout3a3b(self, a: int, b: int) -> str:
        res = ''
        while a or b:
            if len(res) >= 2 and res[-1] == res[-2]:
                writeA = res[-1] == 'b'
            else:
                writeA = a >= b 
            if writeA:
                res += 'a'
                a -= 1
            else:
                res += 'b'
                b -= 1
        return res 
```

#### 767. Reorganize String

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        n = len(s)
        a = Counter(s).most_common()
        m = a[0][1]
        if m > n - m + 1:
            return ''

        res = [''] * n 
        i = 0
        for c, cnt in a:
            for _ in range(cnt):
                res[i] = c 
                i += 2 
                if i >= n:
                    i = 1 
        return ''.join(res)
```

#### 1054. Distant Barcodes

```python
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        n = len(barcodes)
        a = Counter(barcodes).most_common()
        res = [0] * n 
        i = 0
        for v, cnt in a:
            for _ in range(cnt):
                res[i] = v 
                i += 2
                if i >= n:
                    i = 1
        return res 
```

### 414. Third Maximum Number

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        pq = []
        for n in set(nums):
            heappush(pq, -n)
        
        if len(pq) < 3:
            return -pq[0]

        for i in range(3):
            res = -heappop(pq)
        return res
```

- O(n) voting

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        nums = list(set(nums))
        if len(nums) < 3:
            return max(nums)
        first = second = third = -inf 
        for n in nums:
            if n > first:
                first, second, third = n, first, second 
            elif n > second:
                second, third = n, second 
            elif n > third:
                third = n 
        return third 
```

### 2974. Minimum Number Game

```python
class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        heapify(nums)
        res = []
        while nums:
            a, b = heappop(nums), heappop(nums)
            res.extend([b, a])
        return res
```

### 1046. Last Stone Weight

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-n for n in stones]
        heapify(stones)
        while len(stones) > 1:
            a, b = heappop(stones), heappop(stones)
            if a != b:
                heappush(stones, -abs(a - b))
        return -stones[0] if stones else 0
```

### 3264. Final Array State After K Multiplication Operations I

```python
class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        nums = [(n, i) for i, n in enumerate(nums)]
        heapify(nums)
        for _ in range(k):
            val, idx = heappop(nums)
            num = val * multiplier
            heappush(nums, (num, idx))
        nums.sort(key = lambda x: x[1])
        return [n for n, i in nums]
```

### 2558. Take Gifts From the Richest Pile

```python
class Solution:
    def pickGifts(self, gifts: List[int], k: int) -> int:
        gifts = [-n for n in gifts]
        heapify(gifts)
        for _ in range(k):
            num = heappop(gifts)
            heappush(gifts, -floor(sqrt(-num)))
        return -sum(gifts)
```

### 2336. Smallest Number in Infinite Set

```python
class SmallestInfiniteSet:

    def __init__(self):
        self.smallest = 1
        self.pq = []

    def popSmallest(self) -> int:
        res = 0
        if self.pq:
            res = heappop(self.pq)
        else:
            res = self.smallest
            self.smallest += 1
        return res

    def addBack(self, num: int) -> None:
        if num < self.smallest and num not in self.pq: # forget about num not in self.pq
            heappush(self.pq, num)

from sortedcontainers import SortedSet
class SmallestInfiniteSet:

    def __init__(self):
        self.sl = SortedList()
        self.cur = 0

    def popSmallest(self) -> int:
        if self.sl:
            n = self.sl[0]
            self.sl.remove(n)
            return n 
        self.cur += 1
        return self.cur

    def addBack(self, num: int) -> None:
        if num <= self.cur and num not in self.sl:
            self.sl.add(num)
```

### 2530. Maximal Score After Applying K Operations

```python
class Solution:
    def maxKelements(self, nums: List[int], k: int) -> int:
        nums = [-n for n in nums]
        heapify(nums)
        res = 0
        for i in range(k):
            n = -heappop(nums)
            res += n 
            heappush(nums, -ceil(n / 3))
        return res
```

### 3066. Minimum Operations to Exceed Threshold Value II

```python
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # nums = [2,11,10,1,3], k = 10
        heapify(nums)
        res = 0
        while len(nums) >= 2:
            n1, n2 = heappop(nums), heappop(nums)
            if n1 >= k:
                return res 
            res += 1
            ans = min(n1, n2) * 2 + max(n1, n2)
            heappush(nums, ans)
        return res
```

### 1962. Remove Stones to Minimize the Total

```python
class Solution:
    def minStoneSum(self, piles: List[int], k: int) -> int:
        nums = [-n for n in piles]
        heapify(nums)
        for _ in range(k):
            num = heappop(nums)
            heappush(nums, num + floor(-num / 2))
        return -sum(nums)
```

### 703. Kth Largest Element in a Stream

```python
from sortedcontainers import SortedList
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.sl = SortedList(nums)

    def add(self, val: int) -> int:
        self.sl.add(val)
        return self.sl[-self.k]

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.nums = nums
        heapify(self.nums)
        self.k = k 

    def add(self, val: int) -> int:
        heappush(self.nums, val)
        while len(self.nums) > self.k:
            heappop(self.nums)
        return self.nums[0]
```

### 3275. K-th Nearest Obstacle Queries

```python
class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        n = len(queries)
        res = [-1] * n 
        maxHeap = []
        for i, (x, y) in enumerate(queries):
            heappush(maxHeap, -(abs(x) + abs(y)))
            while len(maxHeap) > k:
                heappop(maxHeap)
            if len(maxHeap) == k:
                res[i] = -maxHeap[0]
        return res 
```

### 973. K Closest Points to Origin

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key = lambda x: x[0] ** 2 + x[1] ** 2)
        return points[:k]
```

### 215. Kth Largest Element in an Array

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums = [-n for n in nums]
        heapify(nums)
        for i in range(k):
            v = -heappop(nums)
        return v
```

#### 347. Top K Frequent Elements

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        d = Counter(nums)
        pq = []
        for key, val in d.items():
            heappush(pq, (-val, key))
        
        res = []
        for i in range(k):
            res.append(heappop(pq)[1])
        return res
```

#### 692. Top K Frequent Words

```python
lass Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        pq = []
        for key, val in Counter(words).items():
            heappush(pq, (-val, key))
        
        res = []
        for i in range(k):
            v2, k2 = heappop(pq)
            res.append(k2)
        return res
```

### 1953. Maximum Number of Weeks for Which You Can Work

```python
class Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:
        s = sum(milestones)
        m = max(milestones)
        return (s - m) * 2 + 1 if m > s - m + 1 else s
```

### 2974. Minimum Number Game

```python
class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        heapify(nums)
        res = []
        while nums:
            a, b = heappop(nums), heappop(nums)
            res.extend([b, a])
        return res
```

### 1845. Seat Reservation Manager

```python
class SeatManager:

    def __init__(self, n: int):
        self.nums = list(range(1, n + 1))
        heapify(self.nums)

    def reserve(self) -> int:
        num = heappop(self.nums)
        return num

    def unreserve(self, seatNumber: int) -> None:
        heappush(self.nums, seatNumber)
```

### 2208. Minimum Operations to Halve Array Sum

```python
class Solution:
    def halveArray(self, nums: List[int]) -> int:
        reduced, total = 0, sum(nums)
        res = 0
        nums = [-n for n in nums]
        heapify(nums)
        while nums:
            num = -heappop(nums)
            reduced += num / 2
            res += 1
            heappush(nums, -num / 2)
            if reduced >= total / 2:
                return res 
```

### 2233. Maximum Product After K Increments

```python
class Solution:
    def maximumProduct(self, nums: List[int], k: int) -> int:
        mod = 10 ** 9 + 7
        heapify(nums)
        for _ in range(k):
            n = heappop(nums)
            heappush(nums, n + 1)
        res = 1
        for p in nums:
            res = (res * p) % mod
        return res
```

### 3296. Minimum Number of Seconds to Make Mountain Height Zero

```python
class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        res = 0
        workerTimes = [(unit, unit, 1) for unit in workerTimes]
        heapify(workerTimes)
        while mountainHeight > 0:
            t, unit, cnt = heappop(workerTimes)
            mountainHeight -= 1
            res = max(res, t)
            cnt += 1
            heappush(workerTimes, (t + unit * cnt, unit, cnt))
        return res
```

### 1942. The Number of the Smallest Unoccupied Chair

```python
class Solution:
    def smallestChair(self, times: List[List[int]], targetFriend: int) -> int:
        n = len(times)
        times = [(arrive, leave, i) for i, (arrive, leave) in enumerate(times)]
        times.sort()
        empty = list(range(n)) # index
        heapify(empty)
        used = [] # leave time, index
        for arrive, leave, i in times:
            # free all used chairs whose leave time is <= arrive time
            while used and used[0][0] <= arrive:
                _, idx = heappop(used)
                heappush(empty, idx)
            cur = heappop(empty)
            if i == targetFriend:
                return cur 
            heappush(used, (leave, cur))
```

### 2406. Divide Intervals Into Minimum Number of Groups

```python
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        leaving = []
        for arrival, leave in intervals:
            if leaving and arrival > leaving[0]:
                heapreplace(leaving, leave)
            else:
                heappush(leaving, leave)
        return len(leaving)
```


### 253. Meeting Rooms II

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        pq = []
        for start, end in intervals:
            if pq and start >= pq[0]:
                heapreplace(pq, end)
            else:
                heappush(pq, end)
        return len(pq)
```

### 1167. Minimum Cost to Connect Sticks

```python
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        heapify(sticks)
        res = 0
        while len(sticks) > 1:
            a, b = heappop(sticks), heappop(sticks)
            res += a + b 
            heappush(sticks, a + b)
        return res
```

### 1801. Number of Orders in the Backlog

```python
class Solution:
    def getNumberOfBacklogOrders(self, orders: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        buy, sell = [], []
        for p, a, t in orders:
            if t == 0:
                sold = False
                while sell and sell[0][0] <= p:
                    x, y = heappop(sell)
                    if a > y:
                        a -= y
                    else:
                        sold = True
                        heappush(sell, (x, y - a))
                        break
                if not sold:
                    heappush(buy, (-p, a))
            else:
                bought = False
                while buy and -buy[0][0] >= p:
                    x, y = heappop(buy)
                    if a > y:
                        a -= y
                    else:
                        bought = True
                        heappush(buy, (x, y - a))
                        break
                if not bought:
                    heappush(sell, (p, a))
        return sum(v[1] for v in buy + sell) % mod 
```

### 2462. Total Cost to Hire K Workers

```python
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        n = len(costs)
        if 2 * candidates + k > n:
            costs.sort()
            return sum(costs[:k])
        pre, suf = costs[:candidates], costs[-candidates:]
        heapify(pre)
        heapify(suf)
        res = 0
        i, j = candidates, n - 1 - candidates
        for _ in range(k):
            if pre[0] <= suf[0]:
                res += heapreplace(pre, costs[i])
                i += 1
            else:
                res += heapreplace(suf, costs[j])
                j -= 1
        return res 
```

### 1834. Single-Threaded CPU

```python
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        tasks = [[s, d, i] for i, (s, d) in enumerate(tasks)]
        tasks.sort(key = lambda x: x[0])
        res, pq = [], []
        i, time = 0, tasks[0][0]
        while pq or i < len(tasks):
            while i < len(tasks) and time >= tasks[i][0]:
                heappush(pq, tasks[i][1:])
                i += 1
            if not pq:
                time = max(time, tasks[i][0])
            else:
                procTime, index = heappop(pq)
                time += procTime 
                res.append(index)
        return res 
```

### 1792. Maximum Average Pass Ratio

```python
class Solution:
    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        pq = []
        for x, y in classes:
            heappush(pq, (-((x + 1) / (y + 1) - (x / y)), x, y))
        res = sum(x / y for x, y in classes)
        for _ in range(extraStudents):
            ratio, x, y = heappop(pq)
            res += -ratio
            x, y = x + 1, y + 1
            heappush(pq, (-((x + 1) / (y + 1) - (x / y)), x, y))
        return res / len(classes)
```

### 2931. Maximum Spending After Buying Items

```python
class Solution:
    def maxSpending(self, values: List[List[int]]) -> int:
        pq = []
        for val in values:
            for v in val:
                heappush(pq, v)
        
        day, res = 1, 0
        while pq:
            n = heappop(pq)
            res += day * n 
            day += 1
        return res 
```

### 1882. Process Tasks Using Servers

```python
class Solution:
    def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
        res = []
        pq = []
        for i, s in enumerate(servers):
            heappush(pq, (s, i))
        used = []
        for base, task in enumerate(tasks):
            while used and used[0][0] <= base:
                _, s, i = heappop(used)
                heappush(pq, (s, i))
            if pq:
                s, i = heappop(pq)
                res.append(i)
                heappush(used, (base + task, s, i))
            else:
                t, s, i = heappop(used)
                res.append(i)
                heappush(used, (t + task, s, i))
        return res 
```

### 2402. Meeting Rooms III

```python
class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        idle, used = list(range(n)), []
        res = []
        meetings.sort()
        for s, e in meetings:
            while used and used[0][0] <= s:
                heappush(idle, heappop(used)[1])
            if idle:
                i = heappop(idle)
            else:
                end, i = heappop(used)
                e = end + e - s
            heappush(used, (e, i))
            res.append(i)
        d = Counter(res)
        ans = 0 
        for i in sorted(d.keys()):
            if d[i] > d[ans]:
                ans = i 
        return ans
```