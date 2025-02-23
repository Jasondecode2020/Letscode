## 1 Minimum value(10)

* [1283. Find the Smallest Divisor Given a Threshold](#1283-find-the-smallest-divisor-given-a-threshold)
* [2187. Minimum Time to Complete Trips](#2187-minimum-time-to-complete-trips)
* [1870. Minimum Speed to Arrive on Time](#1870-minimum-speed-to-arrive-on-time)
* [1011. Capacity To Ship Packages Within D Days](#1011-capacity-to-ship-packages-within-d-days)
* [875. Koko Eating Bananas](#875-koko-eating-bananas)
* [475. Heaters](#475-heaters)
* [2594. Minimum Time to Repair Cars](#2594-minimum-time-to-repair-cars)
* [1482. Minimum Number of Days to Make m Bouquets](#1482-minimum-number-of-days-to-make-m-bouquets)
* [2604. Minimum Time to Eat All Grains](#2604-minimum-time-to-eat-all-grains)
* [2702. Minimum Operations to Make Numbers Non-positive](#2702-minimum-operations-to-make-numbers-non-positive)
* [3296. Minimum Number of Seconds to Make Mountain Height Zero](#3296-minimum-number-of-seconds-to-make-mountain-height-zero)
* [3048. Earliest Second to Mark Indices I]

## 2 Maximum Value

* [274. H-Index](#274-h-index)
* [275. H-Index II](#275-h-index-ii)
* [2226. Maximum Candies Allocated to K Children](#2226-maximum-candies-allocated-to-k-children)
* [1898. Maximum Number of Removable Characters](#1898-maximum-number-of-removable-characters)
* [1642. Furthest Building You Can Reach](#1642-furthest-building-you-can-reach)
* [2861. Maximum Number of Alloys](#2861-maximum-number-of-alloys)
* [2258. Escape the Spreading Fire](#2258-escape-the-spreading-fire)
* [1891. Cutting Ribbons](#1891-cutting-ribbons)

## 3 minimize max value
* [410. Split Array Largest Sum](#410)
* [2064. Minimized Maximum of Products Distributed to Any Store](#2064-minimized-maximum-of-products-distributed-to-any-store)
* [1760. Minimum Limit of Balls in a Bag](#1760-minimum-limit-of-balls-in-a-bag)
* [1631. Path With Minimum Effort](#1631-Path-With-Minimum-Effort)
* [2439. Minimize Maximum of Array](#2439-minimize-maximum-of-array)
* [2560. House Robber IV](#2560-house-robber-iv)
* [778. Swim in Rising Water](#778-swim-in-rising-water)
* [2616. Minimize the Maximum Difference of Pairs](#2616-minimize-the-maximum-difference-of-pairs)

## 4 maximize min value

* [1552. Magnetic Force Between Two Balls](#1552-magnetic-force-between-two-balls)
* [2517. Maximum Tastiness of Candy Basket](#2517-maximum-tastiness-of-candy-basket)
* [2812. Find the Safest Path in a Grid](#2812-find-the-safest-path-in-a-grid)
* [1102. Path With Maximum Minimum Value](#1102-path-with-maximum-minimum-value)
* [1231. Divide Chocolate](#1231-divide-chocolate)
* [2528. Maximize the Minimum Powered City](#2528-maximize-the-minimum-powered-city)

## 5 kth smallest/largest

* [378. Kth Smallest Element in a Sorted Matrix](#378-kth-smallest-element-in-a-sorted-matrix)
* [668. Kth Smallest Number in Multiplication Table](#668-kth-smallest-number-in-multiplication-table)
* [719. Find K-th Smallest Pair Distance](#719-find-k-th-smallest-pair-distance)
* [878. Nth Magical Number](#878-nth-magical-number)
* [1201. Ugly Number III](#1201-ugly-number-iii)

* [373. Find K Pairs with Smallest Sums](#373-find-k-pairs-with-smallest-sums)
* [2702. Minimum Operations to Make Numbers Non-positive](#2702-minimum-operations-to-make-numbers-non-positive)

### 1283. Find the Smallest Divisor Given a Threshold

```python
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        def check(m):
            count = 0
            for n in nums:
                count += ceil(n / m)
            return count <= threshold
            
        l, r, res = 1, max(nums), 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```

### 2187. Minimum Time to Complete Trips

```python
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        def check(threshold):
            count = 0
            for t in time:
                count += threshold // t
            return count >= totalTrips
            
        l, r, res = 1, max(time) * totalTrips, 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1870. Minimum Speed to Arrive on Time

```python
class Solution:
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        n = len(dist)
        def check(threshold):
            count = 0
            for i, d in enumerate(dist):
                if i < n - 1:
                    count += ceil(d / threshold)
                else:
                    count += d / threshold
            return count <= hour

        l, r, res = 1, 10 ** 7, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res if res != 0 else -1
```


### 1011. Capacity To Ship Packages Within D Days

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def check(threshold):
            count, total = 0, 0
            for w in weights:
                total += w
                if total > threshold:
                    count += 1
                    total = w
            if total > threshold:
                return False
            return count + 1 <= days

        l, r, res = max(weights), sum(weights), max(weights)
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 875. Koko Eating Bananas

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def check(threshold):
            count = 0
            for p in piles:
                if p % threshold == 0:
                    count += p // threshold
                else:
                    count += p // threshold + 1
            return count <= h

        l, r, res = 1, max(piles), 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 475. Heaters

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        res = -inf
        heaters = [-inf] + heaters + [inf]
        heaters.sort()
        for h in houses:
            idx = bisect_left(heaters, h)
            ans = min(heaters[idx] - h, h - heaters[idx - 1])
            res = max(ans, res)
        return res
```

### 2594. Minimum Time to Repair Cars

```python
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        def check(threshold):
            count = 0
            for r in ranks:
                count += int(sqrt((threshold / r)))
            return count >= cars

        l, r, res = 1, 10 ** 14, 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1482. Minimum Number of Days to Make m Bouquets

```python
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        def valid(res):
            i = 0
            ans = []
            while i < len(res):
                start = i 
                j = start 
                while j < len(res) and res[j] == res[start]:
                    j += 1
                if res[start] == 1:
                    ans.append(j - i)
                i = j
            count = 0
            for n in ans:
                count += n // k
            return count >= m
                    
        def check(threshold):
            n = len(bloomDay)
            res = [0] * n 
            for i, b in enumerate(bloomDay):
                if b <= threshold:
                    res[i] = 1
            return valid(res)
            
        l, r, res = 1, max(bloomDay), -1
        while l <= r:
            mid = l + (r - l) // 2
            if check(mid):
                res = mid 
                r = mid - 1
            else:
                l = mid + 1
        return res
```

### 2604. Minimum Time to Eat All Grains

```python
class Solution:
    def minimumTime(self, hens: List[int], grains: List[int]) -> int:
        hens.sort()
        grains.sort()
        def check(threshold):
            i, j = 0, 0
            while i < len(hens):
                left, right = hens[i], hens[i]
                while j < len(grains):
                    left, right = min(left, grains[j]), max(right, grains[j])
                    if min(hens[i] - left + right - left, right - hens[i] + right - left) <= threshold:
                        j += 1
                    else:
                        break
                i += 1
            return j == len(grains)
        # n * log(r - l)
        l, r, res = 0,  10 ** 9 + 5 * 10 ** 8, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 2702. Minimum Operations to Make Numbers Non-positive

```python
class Solution:
    def minOperations(self, nums: List[int], x: int, y: int) -> int:
        def check(threshold):
            arr = [n - y * threshold for n in nums]
            count = 0
            for n in arr:
                if n > 0:
                    count += ceil(n / (x - y))
            return count <= threshold
            
        l, r, res = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
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
            res = max(res, t)
            mountainHeight -= 1
            cnt += 1
            heappush(workerTimes, (t + unit * cnt, unit, cnt))
        return res 
```

### 274. H-Index

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        def check(threshold):
            count = 0
            for n in citations:
                if n >= threshold:
                    count += 1
            return count >= threshold
            
        l, r, res = 0, max(citations), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 275. H-Index II

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        def check(threshold):
            i = bisect_left(citations, threshold)
            return len(citations) - i >= threshold
            
        l, r, res = 0, 1000, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```


### 2226. Maximum Candies Allocated to K Children

```python
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        def check(threshold):
            count = 0
            if threshold == 0:
                return True
            for c in candies:
                count += c // threshold
            return count >= k

        l, r, res = 0, max(candies), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 1898. Maximum Number of Removable Characters

```python
class Solution:
    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
        def isSub(s, p):
            i, j = 0, 0
            while i < len(s) and j < len(p):
                if p[j] == s[i]:
                    i += 1
                    j += 1
                else:
                    i += 1
            return j == len(p)

        def check(threshold):
            res = ''
            seen = set(removable[:threshold])
            for i, c in enumerate(s):
                if i not in seen:
                    res += c 
            return isSub(res, p)

        l, r, res = 0, len(removable), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```


### 1642. Furthest Building You Can Reach

```python
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        def check(threshold):
            res = heights[:threshold + 1]
            ans = []
            for i in range(1, len(res)):
                if res[i] > res[i - 1]:
                    ans.append(res[i] - res[i - 1])
            ans.sort(reverse = True)
            ans = ans[ladders:]
            return sum(ans) <= bricks

        l, r, res = 0, len(heights) - 1, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2861. Maximum Number of Alloys

```python
class Solution:
    def maxNumberOfAlloys(self, n: int, k: int, budget: int, composition: List[List[int]], stock: List[int], cost: List[int]) -> int:
        # n = 3, k = 2, budget = 15, composition = [[1,1,1],[1,1,10]], stock = [0,0,100], cost = [1,2,3]
        def check(threshold, c):
            needed = [i * threshold for i in c]
            needed = [needed[i] - stock[i] for i, v in enumerate(needed)]
            costs = sum([needed[i] * cost[i] for i in range(len(cost)) if needed[i] > 0])
            return costs <= budget
            
        res = 0
        for c in composition:
            l, r, ans = 0, 10 ** 14, 0
            while l <= r:
                m = l + (r - l) // 2
                if check(m, c):
                    ans = m
                    l = m + 1
                else:
                    r = m - 1
            res = max(res, ans)
        return res
```


### 2258. Escape the Spreading Fire

```python
class Solution:
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        def bfs():
            q = deque()
            for r in range(R):
                for c in range(C):
                    if grid[r][c] == 1:
                        q.append((r, c, 0))
                        fireTime[r][c] = 0
            
            while q:
                r, c, t = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C and grid[row][col] != 2 and fireTime[row][col] == inf:
                        q.append((row, col, t + 1))
                        fireTime[row][col] = t + 1

        def check(stayTime):
            visited = set((0, 0))
            q = deque([(0, 0, stayTime)])
            while q:
                r, c, t = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    if 0 <= row < R and 0 <= col < C and grid[row][col] != 2 and (row, col) not in visited:
                        if row == R - 1 and col == C - 1:
                            return fireTime[row][col] >= t + 1
                        if fireTime[row][col] > t + 1:
                            q.append((row, col, t + 1))
                            visited.add((row, col))
            return False

        R, C = len(grid), len(grid[0])
        fireTime = [[inf] * C for r in range(R)]
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        bfs()
        l, r, res = 0, R * C, -1
        while l <= r:
            mid = l + (r - l) // 2
            if check(mid):
                res = mid
                l = mid + 1
            else:
                r = mid - 1
        return res if res < R * C else 10 ** 9
```

### 1891. Cutting Ribbons

```python
class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        def check(threshold):
            count = 0
            for r in ribbons:
                count += r // threshold
            return count >= k

        l, r, res = 1, 10 ** 5, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2137. Pour Water Between Buckets to Make Water Levels Equal

```python
class Solution:
    def equalizeWater(self, buckets: List[int], loss: int) -> float:
        def check(threshold):
            res = 0
            for b in buckets:
                if b > threshold:
                    res += (b - threshold) * (100 - loss) / 100
            for b in buckets:
                if b < threshold:
                    res -= (threshold - b)
            return res >= 0
        l, r, res = 0, 10 ** 5, 0
        eps = 10 ** -6
        while l <= r:
            m = l + (r - l) / 2
            if check(m):
                res = m 
                l = m + eps
            else:
                r = m - eps
        return res 
```

### 2064. Minimized Maximum of Products Distributed to Any Store

```python
class Solution:
    def minimizedMaximum(self, n: int, quantities: List[int]) -> int:
        def check(threshold):
            count = 0
            for q in quantities:
                count += ceil(q / threshold)
            return count <= n 

        l, r, res = 1, 10 ** 5, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```


### 1760. Minimum Limit of Balls in a Bag

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        def check(threshold):
            count = 0
            for n in nums:
                count += (n - 1) // threshold
            return count <= maxOperations

        l, r, res = 1, 10 ** 9, 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```


### 1631. Path With Minimum Effort

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def minimumEffortPath(self, grid: List[List[int]]) -> int:
        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    for dr, dc in [[1, 0], [0, 1]]:
                        row, col = r + dr, c + dc
                        if 0 <= row < R and 0 <= col < C and abs(grid[row][col] - grid[r][c]) <= threshold and not uf.connected(row * C + col, r * C + c):
                            uf.union(row * C + col, r * C + c)
            return uf.connected(0, (R - 1) * C + C - 1)

        R, C = len(grid), len(grid[0])
        l, r, res = 0, 10 ** 6, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```


### 2439. Minimize Maximum of Array

```python
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        def check(threshold):
            carry = 0
            for i in range(n - 1, 0, -1):
                if nums[i] + carry > threshold:
                    carry += nums[i] - threshold
                else:
                    carry = 0
            return nums[0] + carry <= threshold

        l, r, res = min(nums), max(nums), min(nums)
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```

### 2560. House Robber IV

```python
class Solution:
    def minCapability(self, nums: List[int], k: int) -> int:
        def check(threshold):
            count, i = 0, 0
            while i < len(nums):
                if nums[i] <= threshold:
                    count += 1
                    i += 1
                i += 1
            return count >= k

        l, r, res = 0, max(nums), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 778. Swim in Rising Water

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    if r == 0 and c == 0 and grid[r][c] > threshold:
                        return False # speed up a bit
                    if grid[r][c] <= threshold:
                        for dr, dc in [[1, 0], [0, 1]]:
                            row, col = r + dr, c + dc
                            if 0 <= row < R and 0 <= col < C and grid[row][col] <= threshold and not uf.connected(row * C + col, r * C + c):
                                uf.union(row * C + col, r * C + c)
            return uf.connected(0, (R - 1) * C + C - 1)

        R, C = len(grid), len(grid[0])
        l, r, res = 0, R * C, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```


### 2616. Minimize the Maximum Difference of Pairs

```python
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        def check(threshold):
            count = i = 0
            while i < n - 1:
                if nums[i + 1] - nums[i] <= threshold:
                    i += 2
                    count += 1
                else:
                    i += 1
            return count >= p

        nums.sort()
        n = len(nums)
        l, r, res = 0, nums[-1] - nums[0], 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```


### 1552. Magnetic Force Between Two Balls

```python
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        def check(threshold):
            pre = position[0]
            count = 1
            for i in range(1, len(position)):
                if position[i] - pre >= threshold:
                    pre = position[i]
                    count += 1
            return count >= m 

        position.sort()
        l, r, res = 1, position[-1] - position[0], 1
        while l <= r:
            mid = l + (r - l) // 2
            if check(mid):
                res = mid
                l = mid + 1
            else:
                r = mid - 1
        return res
```

### 2517. Maximum Tastiness of Candy Basket

```python
class Solution:
    def maximumTastiness(self, price: List[int], k: int) -> int:
        def check(threshold):
            count = 1
            pre = price[0]
            for i in range(1, len(price)):
                if price[i] - pre >= threshold:
                    count += 1
                    pre = price[i]
            return count >= k

        price.sort()
        l, r, res = 0, price[-1] - price[0], 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                l = m + 1
            else:
                r = m - 1
        return res
```


### 2812. Find the Safest Path in a Grid

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dist = [[0] * C for r in range(R)]
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        q, visited = deque(), set()
        for r in range(R):
            for c in range(C):
                if grid[r][c]:
                    q.append((r, c, 0))
                    visited.add((r, c))
        while q:
            r, c, d = q.popleft()
            dist[r][c] = d 
            for dr, dc in  directions:
                row, col = r + dr, c + dc 
                if 0 <= row < R and 0 <= col < C and (row, col) not in visited:
                    visited.add((row, col))
                    q.append((row, col, d + 1))

        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    if r == 0 and c == 0 and dist[r][c] < threshold:
                        return False # speed up a bit
                    if dist[r][c] >= threshold:
                        for dr, dc in [[1, 0], [0, 1]]:
                            row, col = r + dr, c + dc
                            if 0 <= row < R and 0 <= col < C and dist[row][col] >= threshold and not uf.connected(row * C + col, r * C + c):
                                uf.union(row * C + col, r * C + c)
            return uf.connected(0, R * C - 1)
        
        l, r, res = 0, R * C, 0
        if R == C == 1 and grid[0][0] == 1:
            return 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                l = m + 1
            else:
                r = m - 1
        return res
```

### 1102. Path With Maximum Minimum Value

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def connected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

class Solution:
    def maximumMinimumPath(self, grid: List[List[int]]) -> int:
        def check(threshold):
            uf = UF(R * C)
            for r in range(R):
                for c in range(C):
                    if r == 0 and c == 0 and grid[r][c] < threshold:
                        return False # speed up a bit
                    if grid[r][c] >= threshold:
                        for dr, dc in [[1, 0], [0, 1]]:
                            row, col = r + dr, c + dc
                            if 0 <= row < R and 0 <= col < C and grid[row][col] >= threshold and not uf.connected(row * C + col, r * C + c):
                                uf.union(row * C + col, r * C + c)
            return uf.connected(0, (R - 1) * C + C - 1)

        R, C = len(grid), len(grid[0])
        l, r, res = 0, max(max(item) for item in grid), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                l = m + 1
            else:
                r = m - 1
        return res
```

### 1231. Divide Chocolate

```python
class Solution:
    def maximizeSweetness(self, sweetness: List[int], k: int) -> int:
        def check(threshold, k):
            i, count, total = 0, 0, 0
            while i < len(sweetness):
                total += sweetness[i]
                if total >= threshold:
                    total = 0
                    count += 1
                i += 1
            return count >= k + 1
            
        l, r, res = 0, sum(sweetness) // (k + 1), 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m, k):
                res = m
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2528. Maximize the Minimum Powered City

```python
class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        n = len(stations)
        pre = list(accumulate(stations, initial = 0))
        for i in range(n):
            stations[i] = pre[min(i + r + 1, n)] - pre[max(i - r, 0)]

        def check(threshold):
            pre = extra_stations = 0
            f = [0] * (n + 1)
            for i, power in enumerate(stations):
                pre += f[i]
                station_num = threshold - power - pre
                if station_num > 0:
                    extra_stations += station_num 
                    if extra_stations > k:
                        return False 
                    pre += station_num
                    if (i + 2 * r + 1) < n:
                        f[i + 2 * r + 1] -= station_num
            return True

        left, right, res = 0, 10 ** 20, 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                res = mid 
                left = mid + 1
            else:
                right = mid - 1
        return res
```

### 378. Kth Smallest Element in a Sorted Matrix

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        def check(threshold):
            r, c, count = R, 1, 0
            while r >= 1 and c <= C:
                if matrix[r - 1][c - 1] <= threshold:
                    count += r 
                    c += 1
                else:
                    r -= 1
            return count >= k 
            
        R, C = len(matrix), len(matrix[0])
        l, r, res = -10 ** 9, 10 ** 9, 1
        while l <= r:
            m = (l + r) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res 
```

### 668. Kth Smallest Number in Multiplication Table

```python
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        def check(threshold):
            r, c, count = R, 1, 0
            while r >= 1 and c <= C:
                if r * c <= threshold:
                    count += r 
                    c += 1
                else:
                    r -= 1
            return count >= k 
            
        R, C = m, n 
        l, r, res = 1, R * C, 1
        while l <= r:
            m = (l + r) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res 
```

### 719. Find K-th Smallest Pair Distance

```python 
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        n = len(nums)
        def check(threshold):
            count = 0
            for i, x in enumerate(nums):
                count += bisect_right(nums, x + threshold, i + 1, n) - (i + 1)
            return count >= k
        nums.sort()
        l, r, res = 0, max(nums) - min(nums), 0
        while l <= r:
            m = (l + r) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res 
```

### 878. Nth Magical Number

```python 
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        def check(threshold):
            count = 0
            count += threshold // a 
            count += threshold // b 
            count -= threshold // lcm(a, b)
            return count >= n 

        l, r, res = 2, 10 ** 18, 2
        mod = 10 ** 9 + 7
        while l <= r:
            m = (l + r) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res % mod
```

### 373. Find K Pairs with Smallest Sums

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        pq = [(nums1[0] + nums2[0], 0, 0)]
        s = set((0, 0))
        res = []
        for i in range(k):
            n, a, b = heappop(pq)
            res.append([nums1[a], nums2[b]])
            if (a + 1, b) not in s and a + 1 < len(nums1):
                s.add((a + 1, b))
                heappush(pq, (nums1[a + 1] + nums2[b], a + 1, b))
            if (a, b + 1) not in s and b + 1 < len(nums2):
                s.add((a, b + 1))
                heappush(pq, (nums1[a] + nums2[b + 1], a, b + 1))
        return res
```

### 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

```python
class Solution:
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        R, C = len(mat), len(mat[0])
        dp = [[0] * (C + 1) for r in range(R + 1)]
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1] - dp[r - 1][c - 1] + mat[r - 1][c - 1]
                
        def check(m):
            for r in range(m, R + 1):
                for c in range(m, C + 1):
                    s = dp[r][c] - dp[r - m][c] - dp[r][c - m] + dp[r - m][c - m]
                    if s <= threshold:
                        return True
            return False

        l, r, res = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2702. Minimum Operations to Make Numbers Non-positive

```python
class Solution:
    def minOperations(self, nums: List[int], x: int, y: int) -> int:
        def check(threshold):
            arr = [n - threshold * y for n in nums]
            count = 0
            for n in arr:
                if n > 0:
                    count += ceil(n / (x - y))
            return count <= threshold
            
        l, r, res = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res 
```

### 1060. Missing Element in Sorted Array

```python
class Solution:
    def missingElement(self, nums: List[int], k: int) -> int:
        def check(threshold):
            j = bisect_left(nums, threshold)
            if j < len(nums) and nums[j] == threshold:
                return threshold - nums[0] - j >= k
            return threshold - nums[0] - (j - 1) >= k
            
        l, r, res = nums[0] + 1, 10 ** 9, nums[0] + 1
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m
                r = m - 1
            else:
                l = m + 1
        return res
```

### 1300. Sum of Mutated Array Closest to Target

```python
class Solution:
    def findBestValue(self, arr: List[int], target: int) -> int:
        def check1(threshold):
            res = 0
            for n in arr:
                if n > threshold:
                    res += threshold
                else:
                    res += n 
            if res >= target:
                self.ans1 = min(self.ans1, res - target)
            return res >= target
        
        def check2(threshold):
            res = 0
            for n in arr:
                if n > threshold:
                    res += threshold
                else:
                    res += n 
            if res < target:
                self.ans2 = min(self.ans2, target - res)
            return res < target


        if sum(arr) <= target:
            return max(arr)

        self.ans1, self.ans2 = inf, inf 
        l, r, res1 = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check1(m):
                res1 = m
                r = m - 1
            else:
                l = m + 1
        
        l, r, res2 = 0, 10 ** 9, 0
        while l <= r:
            m = l + (r - l) // 2
            if check2(m):
                res2 = m
                l = m + 1
            else:
                r = m - 1
        if self.ans1 < self.ans2:
            return res1
        elif self.ans1 > self.ans2:
            return res2
        return min(res1, res2)
```

### 1891. Cutting Ribbons

```python
class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        def check(threshold):
            count = 0
            for r in ribbons:
                count += r // threshold
            return count >= k

        l, r, res = 1, 10 ** 5, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 2137. Pour Water Between Buckets to Make Water Levels Equal

```python
class Solution:
    def equalizeWater(self, buckets: List[int], loss: int) -> float:
        def check(threshold):
            res = 0
            for b in buckets:
                if b > threshold:
                    res += (b - threshold) * (100 - loss) / 100
            for b in buckets:
                if b < threshold:
                    res -= (threshold - b)
            return res >= 0
        l, r, res = 0, 10 ** 5, 0
        eps = 10 ** -6
        while l <= r:
            m = l + (r - l) / 2
            if check(m):
                res = m 
                l = m + eps
            else:
                r = m - eps
        return res 
```

### 774. Minimize Max Distance to Gas Station

```python
class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        def check(threshold):
            count = 0
            for d in dist:
                count += int(d / threshold)
            return count <= k

        dist = [stations[i] - stations[i - 1] for i in range(1, len(stations))]
        l, r, res = 0, max(dist), 0
        while abs(r - l) > 10 ** -6:
            m = l + (r - l) / 2
            if check(m):
                res = m
                r = m
            else:
                l = m
        return res
```

### 878. Nth Magical Number

```python
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        mod = 10 ** 9 + 7
        def check(threshold):
            cnt = 0
            cnt += threshold // a
            cnt += threshold // b
            cnt -= threshold // lcm(a, b)
            return cnt >= n 
        l, r, res = 2, 4 * (10 ** 4) * (10 ** 9), 2
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res % mod
```

### 1201. Ugly Number III

```python
class Solution:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        def check(threshold):
            cnt = 0
            cnt += threshold // a
            cnt += threshold // b
            cnt += threshold // c 
            cnt -= threshold // lcm(a, b)
            cnt -= threshold // lcm(b, c)
            cnt -= threshold // lcm(a, c)
            cnt += threshold // lcm(a, b, c)
            return cnt >= n 
        l, r, res = 1, max(a, b, c) * (10 ** 9), 2
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = m 
                r = m - 1
            else:
                l = m + 1
        return res
```