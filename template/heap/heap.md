## heap

### minHeap

### maxHeap

> same (215, 347, 692)

- * [215. Kth Largest Element in an Array]()
- * [347. Top K Frequent Elements]()
- * [692. Top K Frequent Words]()

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

### 215. Kth Largest Element in an Array

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        pq = []
        for n in nums:
            heappush(pq, -n)
        
        for i in range(k):
            res = -heappop(pq)
        return res
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
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        d = Counter(words)
        pq = []
        for key, val in d.items():
            heappush(pq, (-val, key))
        
        res = []
        for i in range(k):
            res.append(heappop(pq)[1])
        return res
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

### twoHeaps

### regret heap

### 264. Ugly Number II

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        pq, s = [1], set([1])
        for i in range(n - 1):
            num = heappop(pq)
            for factor in [2, 3, 5]:
                if (res := factor * num) not in s:
                    s.add(res)
                    heappush(pq, res)
        return heappop(pq)
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

### 2208. Minimum Operations to Halve Array Sum

```python
class Solution:
    def halveArray(self, nums: List[int]) -> int:
        total = sum(nums)
        pq = [-n for n in nums]
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