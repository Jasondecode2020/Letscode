### 1705. Maximum Number of Eaten Apples

```python
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        d = Counter(barcodes)
        pq = []
        for k, v in d.items():
            heappush(pq, [-v, k])

        res = []
        while len(pq) > 1:
            freq1, first = heappop(pq)
            freq2, second = heappop(pq)
            res.extend([first, second])
            freq1, freq2 = -freq1, -freq2
            if freq1 - 1 > 0:
                heappush(pq, [1 - freq1, first])
            if freq2 - 1 > 0:
                heappush(pq, [1 - freq2, second])
        if pq:
            res.append(pq[0][1])
        return res
```

### 767. Reorganize String

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        c = Counter(s)
        mx = max(c.values())
        if mx > len(s) // 2 + 1 and len(s) % 2:
            return ''
        if mx > len(s) // 2 and len(s) % 2 == 0:
            return ''
        res = ''
        pq = []
        for k, v in c.items():
            heappush(pq, (-v, k))
        while pq:
            v, k = heappop(pq)
            res += k
            if pq:
                v2, k2 = heappop(pq)
                res += k2 
                v2 = -v2
                if v2 - 1 != 0:
                    heappush(pq, (-(v2 - 1), k2))
            v = -v
            if v - 1 != 0:
                heappush(pq, (-(v - 1), k))
        return res
```

### 621. Task Scheduler

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        d = {}
        for i in tasks:
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
        lst = sorted(d.values(), reverse = True)
        max_freq = lst[0]
        count = lst.count(max_freq)
        return max(len(tasks), (max_freq - 1) * (n + 1) + count)
```