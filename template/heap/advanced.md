* [1834. Single-Threaded CPU](#1834-single-threaded-cpu)
* [2931. Maximum Spending After Buying Items](#2931-maximum-spending-after-buying-items)
* [1705. Maximum Number of Eaten Apples](#1705-maximum-number-of-eaten-apples)

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

### 2931. Maximum Spending After Buying Items

```python
class Solution:
    def maxSpending(self, values: List[List[int]]) -> int:
        pq = []
        for val in values:
            for v in val:
                heappush(pq, v)
        cnt = 1
        res = 0
        while pq:
            n = heappop(pq)
            res += n * cnt 
            cnt += 1
        return res 
```

### 1705. Maximum Number of Eaten Apples

```python
class Solution:
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        pq, i, res = [], 0, 0
        while i < len(apples) or pq:
            while pq and pq[0][0] <= i:
                heappop(pq)
            if i < len(apples) and apples[i]:
                heappush(pq, [i + days[i], apples[i]])
            if pq:
                pq[0][1] -= 1
                res += 1
                if not pq[0][1]:
                    heappop(pq)
            i += 1
        return res
```