## heap

* [215. Kth Largest Element in an Array](#215-kth-largest-element-in-an-array)
* [347. Top K Frequent Elements](#347-top-k-frequent-elements)
* [692. Top K Frequent Words](#692-top-k-frequent-words)
* [414. Third Maximum Number](#414-third-maximum-number)
* [973. K Closest Points to Origin](#973-k-closest-points-to-origin)

* [2974. Minimum Number Game](#2974-minimum-number-game)
* [1046. Last Stone Weight](#1046-last-stone-weight)
* [2558. Take Gifts From the Richest Pile](#2558-take-gifts-from-the-richest-pile)
* [1845. Seat Reservation Manager](#1845-seat-reservation-manager)
* [2530. Maximal Score After Applying K Operations](#2530-maximal-score-after-applying-k-operations)

* [1962. Remove Stones to Minimize the Total](#1962-remove-stones-to-minimize-the-total)
* [3066. Minimum Operations to Exceed Threshold Value II](#3066-minimum-operations-to-exceed-threshold-value-ii)
* [2336. Smallest Number in Infinite Set](#2336-smallest-number-in-infinite-set)
* [3264. Final Array State After K Multiplication Operations I](#3264-final-array-state-after-k-multiplication-operations-i)
* [2208. Minimum Operations to Halve Array Sum](#2208-minimum-operations-to-halve-array-sum)

* [2233. Maximum Product After K Increments](#2233-maximum-product-after-k-increments)
* [1942. The Number of the Smallest Unoccupied Chair](#1942-the-number-of-the-smallest-unoccupied-chair)
* [2462. Total Cost to Hire K Workers](#2462-total-cost-to-hire-k-workers)
* [253. Meeting Rooms II](#253-meeting-rooms-ii)
* [1167. Minimum Cost to Connect Sticks](#1167-minimum-cost-to-connect-sticks)


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

### 973. K Closest Points to Origin

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key = lambda x: x[0] ** 2 + x[1] ** 2)
        return points[:k]
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
        pq = []
        for s in stones:
            heappush(pq, -s)
        while len(pq) > 1:
            first, second = heappop(pq), heappop(pq)
            if first != second:
                heappush(pq, -abs(first - second))
        return -pq[0] if pq else 0
```

### 2558. Take Gifts From the Richest Pile

```python
class Solution:
    def pickGifts(self, gifts: List[int], k: int) -> int:
        pq = []
        for g in gifts:
            heappush(pq, -g)

        for i in range(k):
            n = -heappop(pq)
            n = floor(sqrt(n))
            heappush(pq, -n)
        return -sum(pq)
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

### 1962. Remove Stones to Minimize the Total

```python
class Solution:
    def minStoneSum(self, piles: List[int], k: int) -> int:
        pq = []
        for p in piles:
            heappush(pq, -p)

        for i in range(k):
            n = -heappop(pq)
            n = ceil(n / 2)
            heappush(pq, -n)
        return -sum(pq)
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

```

### 3264. Final Array State After K Multiplication Operations I

```python
class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        nums = [(n, i) for i, n in enumerate(nums)]
        heapify(nums)
        for i in range(k):
            n, i = heappop(nums)
            heappush(nums, (n * multiplier, i))
        nums.sort(key = lambda x: x[1])
        return [n for n, i in nums]
```

### 2208. Minimum Operations to Halve Array Sum

```python
class Solution:
    def halveArray(self, nums: List[int]) -> int:
        pq = [-n for n in nums]
        total = sum(nums)
        heapify(pq)
        res, ans = 0, 0
        while pq:
            n = -heappop(pq)
            if ans < total / 2:
                ans += n / 2
                res += 1
                heappush(pq, -n/2)
            else:
                break
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

### 1801. Number of Orders in the Backlog

### 2406. Divide Intervals Into Minimum Number of Groups

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

### 253. Meeting Rooms II

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = []
        for start, end in intervals:
            events.append([start, 1])
            events.append([end, -1])
        events.sort()
        
        count, res = 0, 1
        for time, sign in events:
            count += sign
            res = max(res, count)
        return res
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
