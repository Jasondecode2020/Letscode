## template 1: find minimum value
```python
def fn(arr):
    def check(x):
        # some code
        return BOOLEAN

    l = MINIMUM_POSSIBLE_ANSWER
    r = MAXIMUM_POSSIBLE_ANSWER
    while l <= r:
        m = l + (r - l) // 2
        if check(m):
            r = m - 1
        else:
            l = m + 1
    return l
```

## template 2: find maximum value

```python
def fn(arr):
    def check(x):
        # some code
        return BOOLEAN

    l = MINIMUM_POSSIBLE_ANSWER
    r = MAXIMUM_POSSIBLE_ANSWER
    while l <= r:
        m = (r - l) // 2
        if check(m):
            l = m + 1
        else:
            r = m - 1
    return r
```
* 275. H 指数 II
* 1283. 使结果不超过阈值的最小除数 1542
* 2187. 完成旅途的最少时间 1641
* 2226. 每个小孩最多能分到多少糖果 1646
* 1870. 准时到达的列车最小时速 1676
* 1011. 在 D 天内送达包裹的能力 1725
* 875. 爱吃香蕉的珂珂 1766
* 1898. 可移除字符的最大数目 1913
* 1482. 制作 m 束花所需的最少天数 1946
* 1642. 可以到达的最远建筑 1962
* 2258. 逃离火灾 2347

### 最小化最大值
* 2064. Minimized Maximum of Products Distributed to Any Store 1886
* 1760. Minimum Limit of Balls in a Bag 1940
* 2439. Minimize Maximum of Array 1965
* 2560. House Robber IV 2081
* 778. Swim in Rising Water 2097 相当于最小化路径最大值
* 2616. Minimize the Maximum Difference of Pairs 2155

* 2513. 最小化两个数组中的最大值 2302 later

### 最大化最小值
* 1552. Magnetic Force Between Two Balls 1920
* 2861. 最大合金数 1981
* 2517. Maximum Tastiness of Candy Basket 2021
* 2812. Find the Safest Path in a Grid 2154

* 2528. 最大化城市的最小供电站数目 2236 later

### 第 K 小/大（部分题目也可以用堆解决）
* 378. 有序矩阵中第 K 小的元素
* 373. 查找和最小的 K 对数字
* 719. 找出第 K 小的数对距离
* 1439. 有序矩阵中的第 k 个最小数组和 2134
* 786. 第 K 个最小的素数分数 2169
* 2040. 两个有序数组的第 K 小乘积 2518
* 2386. 找出数组的第 K 大和 2648

三、问题难点
检查函数：检查该值是否复合答案，即定义check()函数

简单加和：最简单的线性扫描
1300. 转变数组后最接近目标值的数组和 no
1891. 割绳子
历史标记：带有历史标记的线性扫描问题，稍微复杂了点
1954. 收集足够苹果的最小花园周长
1802. 有界数组中指定下标处的最大值
暴力匹配：字符串问题
1062. 最长重复子串
隐式最值：有的问题没有直接告诉是求边界最值，需要自己转换成最值问题，难度瞬间加大！

287. 寻找重复数
1648. 销售价值减少的颜色球
274. H 指数
400. 第 N 位数字
特殊二分

浮点二分：将浮点数转为非浮点数便可
2137. 通过倒水操作让所有的水桶所含水量相等
边界未知：定义最大数便可
2187. 完成旅途的最少时间

* 2702. Minimum Operations to Make Numbers Non-positive

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