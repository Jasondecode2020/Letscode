## template 1: regret pq
- continous uf with rank

```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key = lambda x: x[1])
        pq = []
        day = 0
        for d, l in courses:
            if d + day <= l:
                day += d 
                heappush(pq, -d)
            elif pq and d < -pq[0]:
                duration = -heappop(pq)
                heappush(pq, -d)
                day -= duration - d 
        return len(pq)
```

### Questions list

* 1 [630. Course Schedule III](#630-Course-Schedule-III)
* 2 [871. Minimum Number of Refueling Stops](#871-Minimum-Number-of-Refueling-Stops)
* 3 [200. Number of Islands](#200-Number-of-Islands)
* 4 [261. Graph Valid Tree](#261-Graph-Valid-Tree)
* 5 [305. Number of Islands II](#305-Number-of-Islands-II)

### 630. Course Schedule III

```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key = lambda x: x[1])
        pq = []
        day = 0
        for d, l in courses:
            if d + day <= l:
                day += d 
                heappush(pq, -d)
            elif pq and d < -pq[0]:
                duration = -heappop(pq)
                heappush(pq, -d)
                day -= duration - d 
        return len(pq)
```

### 871. Minimum Number of Refueling Stops

```python
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        res, i, pq = 0, 0, []
        while startFuel < target:
            while i < len(stations) and stations[i][0] <= startFuel:
                heappush(pq, -stations[i][1])
                i += 1
            if not pq:
                return -1
            startFuel -= heappop(pq)
            res += 1
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
