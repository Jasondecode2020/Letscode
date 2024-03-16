## sort pq

### 1383. Maximum Performance of a Team

```python
class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        # [2,10,3,1,5,8]
        # [5,4, 3,9,7,2]
        # [(9, 1), (7, 5), (5, 2), (4, 10), (3, 3), (2, 8)]
        # 9 -> 42 -> 35 -> 60
        res, total, mod = 0, 0, 10 ** 9 + 7
        pq = []
        for e, s in sorted(zip(efficiency, speed), reverse = True):
            total += s 
            heappush(pq, s)
            if len(pq) > k:
                mn = heappop(pq)
                total -= mn
            res = max(res, e * total)
        return res % mod
```

### 857. Minimum Cost to Hire K Workers

```python
class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        # [70,50,30]
        # [10,20,5]
        # [7, 2.5, 6]
        nums = []
        for w, q in zip(wage, quality):
            nums.append((w / q, w, q))
        nums.sort()

        pq_quality = []
        res, total = inf, 0
        for r, w, q in nums:
            total += q
            heappush(pq_quality, -q)
            if len(pq_quality) > k:
                mx = -heappop(pq_quality)
                total -= mx 
            if len(pq_quality) == k:
                res = min(res, r * total)
        return res 
```

### 2542. Maximum Subsequence Score

```python
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        res, total, pq = 0, 0, []
        for n1, n2 in sorted(zip(nums1, nums2), key = lambda x: -x[1]):
            total += n1 
            heappush(pq, n1)
            if len(pq) > k:
                mn = heappop(pq)
                total -= mn 
            if len(pq) == k:
                res = max(res, n2 * total)
        return res
```