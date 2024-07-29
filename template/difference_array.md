## difference array(26)

- 26 questions to practise 1d and 2d difference array

### 1d difference array(contigous)(23)

#### Basics(10)

* 1. [2848. Points That Intersect With Cars](#2848-Points-That-Intersect-With-Cars) 1230
* 2. [1893. Check if All the Integers in a Range Are Covered](#1893-Check-if-All-the-Integers-in-a-Range-Are-Covered) 1307
* 3. [495. Teemo Attacking](#495-teemo-attacking) 1350
* 4. [1854. Maximum Population Year](#1854-maximum-population-year) 1370
* 5. [370. Range Addition](#370-Range-Addition) 1400
* 6. [1094. Car Pooling](#1094-car-pooling) 1441
* 7. [1109. Corporate Flight Bookings](#1109-Corporate-Flight-Bookings) 1570
* 8. [252. Meeting Rooms](#252-meeting-rooms) 1600
* 9. [253. Meeting Rooms II](#253-meeting-rooms-ii) 1601
* 10. [2406. Divide Intervals Into Minimum Number of Groups](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups) 1731

#### Advanced 1 (with other conditions)(3)

* 1. [2237. Count Positions on Street With Required Brightness](#2237-count-positions-on-street-with-required-brightness) 1700
* 2. [2381. Shifting Letters II](#2381-Shifting-Letters-II) 1793
* 3. [1589. Maximum Sum Obtained of Any Permutation](#1589-maximum-sum-obtained-of-any-permutation) 1871

#### Advanced 2 (with condition of start)(3)

* 1. [995. Minimum Number of K Consecutive Bit Flips](#995-minimum-number-of-k-consecutive-bit-flips) 1835
* 2. [2772. Apply Operations to Make All Array Elements Equal to Zero](#2772-apply-operations-to-make-all-array-elements-equal-to-zero) 2029
* 3. [2528. Maximize the Minimum Powered City](#2528-maximize-the-minimum-powered-city) 2236

#### Advanced 3 (discrete array)(2)

* 1. [1943. Describe the Painting](#1943-describe-the-painting) 1969
* 2. [2251. Number of Flowers in Full Bloom](#2251-number-of-flowers-in-full-bloom) 2251

#### Advanced 4 (with more difference array calculations)(5)

* 1. [798. Smallest Rotation with Highest Score](#798-smallest-rotation-with-highest-score) 2129
* 2. [1674. Minimum Moves to Make Array Complementary](#1674-minimum-moves-to-make-array-complementary) 2333
* 3. [2015. Average Height of Buildings in Each Segment](#2015-average-height-of-buildings-in-each-segment) 2400
* 4. [3009. Maximum Number of Intersections on the Chart](#3009-maximum-number-of-intersections-on-the-chart) 2500
* 5. [3017. Count the Number of Houses at a Certain Distance II](#3017-count-the-number-of-houses-at-a-certain-distance-ii) 2709

### 2d difference array (3)

* [2536. Increment Submatrices by One](#2536-increment-submatrices-by-one) 1583
* [850. Rectangle Area II](#850-rectangle-area-ii) 2236
* [2132. Stamping the Grid](#2132-stamping-the-grid) 2364

### 2848. Points That Intersect With Cars

```python
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        n = max(e for s, e in nums)
        f = [0] * (n + 2)
        for s, e in nums:
            f[s] += 1
            f[e + 1] -= 1
        return sum(s > 0 for s in accumulate(f))
```

### 1893. Check if All the Integers in a Range Are Covered

```python
class Solution:
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        f = [0] * 52
        for s, e in ranges:
            f[s] += 1
            f[e + 1] -= 1
        return all(v > 0 for i, v in enumerate(accumulate(f)) if left <= i <= right)
```

### 495. Teemo Attacking

```python
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        mx = max(timeSeries) + duration
        nums = [0] * (mx + 1)
        for t in timeSeries:
            nums[t] += 1
            nums[t + duration] -= 1
        nums = list(accumulate(nums))
        return sum([n > 0 for n in nums])
```

### 1854. Maximum Population Year

```python
class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        f = [0] * 2051
        for s, e in logs:
            f[s] += 1
            f[e] -= 1
        pre = list(accumulate(f))
        mx = max(pre)
        for i, n in enumerate(pre):
            if n == mx:
                return i
```

### 370. Range Addition

```python
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        f = [0] * (length + 1)
        for start, end, inc in updates:
            f[start] += inc
            f[end + 1] -= inc
        f = list(accumulate(f))
        return f[: -1]
```

### 1094. Car Pooling

```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        f = [0] * 1001
        for c, s, e in trips:
            f[s] += c
            f[e] -= c
        pre = list(accumulate(f))
        return max(pre) <= capacity
```

### 1109. Corporate Flight Bookings

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        f = [0] * (n + 2)
        for s, e, v in bookings:
            f[s] += v
            f[e + 1] -= v
        pre = list(accumulate(f))
        return pre[1: -1]
```

### 252. Meeting Rooms

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        f = [0] * (10 ** 6 + 1)
        for s, e in intervals:
            f[s] += 1
            f[e] -= 1
        pre = list(accumulate(f))
        return max(pre) <= 1
```

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        events = []
        for start, end in intervals:
            events.append([start, 1])
            events.append([end, -1])
        events.sort()
        
        count = 0
        for time, sign in events:
            count += sign
            if count > 1:
                return False
        return True
```

### 253. Meeting Rooms II

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        f = [0] * (10 ** 6 + 1)
        for s, e in intervals:
            f[s] += 1
            f[e] -= 1
        pre = list(accumulate(f))
        return max(pre)
```

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        events = []
        for s, e in intervals:
            events.append((s, 1))
            events.append((e, -1))
        events.sort()

        res, count = 0, 0
        for x, sign in events:
            if sign == 1:
                count += 1
            else:
                count -= 1
            res = max(res, count)
        return res
```

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        pq = []
        intervals.sort()
        for s, e in intervals:
            if pq and s >= pq[0]:
                heapreplace(pq, e)
            else:
                heappush(pq, e)
        return len(pq)
```


### 2406. Divide Intervals Into Minimum Number of Groups

```python
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        f = [0] * (10 ** 6 + 2)
        for s, e in intervals:
            f[s] += 1
            f[e + 1] -= 1
        pre = list(accumulate(f))
        return max(pre)
```

- heap

```python
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        pq = []
        for s, e in intervals:
            if pq and s > pq[0]:
                heapreplace(pq, e)
            else:
                heappush(pq, e)
        return len(pq)
```

### 2237. Count Positions on Street With Required Brightness

```python
class Solution:
    def meetRequirement(self, n: int, lights: List[List[int]], requirement: List[int]) -> int:
        f = [0] * (n + 1)
        for p, r in lights:
            s, e = max(0, p - r), min(n - 1, p + r)
            f[s] += 1
            f[e + 1] -= 1
        nums = list(accumulate(f[:-1]))
        return sum(n >= r for n, r in zip(nums, requirement))
```

### 3009. Maximum Number of Intersections on the Chart

```python
class Solution:
    def maxIntersectionCount(self, y: List[int]) -> int:
        f = defaultdict(int)
        for a, b in pairwise(y):
            if a < b:
                f[a] += 1
                f[b] -= 1
            else:
                f[b + 0.5] += 1
                f[a + 0.5] -= 1
        last = y[-1]
        f[last] += 1
        f[last + 0.5] -= 1
        arr = sorted([[k, v] for k, v in f.items()])
        for i in range(1, len(arr)):
            arr[i][1] += arr[i - 1][1]
        return max(b for a, b in arr)
```

### 3017. Count the Number of Houses at a Certain Distance II

```python
class Solution:
    def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
        # 调整 x, y 的大小关系
        if x > y: x, y = y, x
        ans = [0] * (n + 1)
        x -= 1
        y -= 1
        for i in range(n):
            # 不需要中介的情况
            if abs(i - x) + 1 > abs(i - y):
                ans[1] += 1
                ans[n - i] -= 1
            # 需要中介的情况
            else:
                # 找到分界点
                d = abs(i - x) + 1
                sep = (y + i + d) // 2
                # 分界点左侧是直接走，距离从 1 到 sep - i
                ans[1] += 1
                ans[sep - i + 1] -= 1
                # 分界点及其右侧与 y 左侧，通过中介，距离从 d + 1 到 d + y - (sep + 1)
                ans[d + 1] += 1
                ans[d + y - sep] -= 1
                # y 及其右侧，距离从 d 到 d + (n - 1) - y
                ans[d] += 1
                ans[d + n - y] -= 1
        ans = list(accumulate(ans))
        return [x * 2 for x in ans[1:]]
```


### 798. Smallest Rotation with Highest Score

```python
class Solution:
    def bestRotation(self, nums: List[int]) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        for i, v in enumerate(nums):
            if i >= v:
                f[0] += 1
                f[i - v + 1] -= 1
                f[i + 1] += 1
            else:
                f[i + 1] += 1
                f[i + 1 + n - v] -= 1
        pre = list(accumulate(f))
        mx = max(pre)
        for i, n in enumerate(pre):
            if n == mx:
                return i 
```

### 1674. Minimum Moves to Make Array Complementary

```python
class Solution:
    def minMoves(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        f = [0] * (2 * limit + 2)
        for i in range(n // 2):
            a = min(nums[i], nums[n - i - 1])
            b = max(nums[i], nums[n - i - 1])
            f[2] += 2
            f[a + 1] -= 1
            f[a + b] -= 1
            f[a + b + 1] += 1
            f[b + limit + 1] += 1
        pre = list(accumulate(f[2:]))
        return min(pre)
```

### 1943. Describe the Painting

```python
class Solution:
    def splitPainting(self, segments: List[List[int]]) -> List[List[int]]:
        d = defaultdict(int)
        for s, e, v in segments:
            d[s] += v
            d[e] -= v
        arr = sorted([[k, v] for k, v in d.items()])
        for i in range(1, len(arr)):
            arr[i][1] += arr[i - 1][1]

        res = []
        for a, b in pairwise(arr):
            k1, v1 = a
            k2, v2 = b 
            if v1:
                res.append([k1, k2, v1])
        return res
```

### 2251. Number of Flowers in Full Bloom

```python
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        d = defaultdict(int)
        for s, e in flowers:
            d[s] += 1
            d[e + 1] -= 1
        pre = sorted([[k, v] for k, v in d.items()])
        for i in range(1, len(pre)):
            pre[i][1] += pre[i - 1][1]
        
        arr = []
        for a, b in pairwise(pre): 
            k1, v1 = a
            k2, v2 = b 
            arr.append([k1, k2 - 1, v1])
        res = []
        for pos in people:
            i = bisect_right(arr, pos, key = lambda x: x[0])
            if i == 0 or (i == len(arr) and pos > arr[i - 1][1]):
                res.append(0)
            else:
                res.append(arr[i - 1][2])
        return res 
```

### 995. Minimum Number of K Consecutive Bit Flips

```python
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        res, flip = 0, 0
        for i, v in enumerate(nums):
            flip += f[i]
            if (flip + v) % 2 == 0:
                if i + k > n:
                    return -1
                flip += 1
                res += 1
                f[i + k] -= 1
        return res 
```

- same idea using queue

```python
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        q = deque()
        n = len(nums)
        res = 0
        for r in range(n):
            if q and r - q[0] + 1 > k:
                q.popleft()
            if (len(q) + nums[r]) % 2 == 0:
                if r + k - 1 >= n:
                    return -1
                res += 1
                q.append(r)
        return res
```


### 2381. Shifting Letters II

```python
class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        n = len(s)
        f = [0] * (n + 1)
        for start, end, direction in shifts:
            if direction == 1:
                f[start] += 1
                f[end + 1] -= 1
            else:
                f[start] += -1
                f[end + 1] -= -1
        pre = list(accumulate(f))[:-1]
        res = ''
        for c, v in zip(list(s), pre):
            v %= 26
            res += chr(ord(c) + v) if ord(c) + v <= ord('z') else chr(ord(c) + v - 26)
        return res
```

### 1589. Maximum Sum Obtained of Any Permutation

- difference array, sorting to find the most frequent
```python
class Solution:
    def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
        n, mod = len(nums), 10 ** 9 + 7
        f = [0] * (n + 1)
        for s, e in requests:
            f[s] += 1
            f[e + 1] -= 1
            
        pre = list(accumulate(f[:-1]))
        nums.sort()
        pre.sort()
        res = 0
        for a, b in zip(nums, pre):
            res += a * b
            res %= mod 
        return res 
```

### 2772. Apply Operations to Make All Array Elements Equal to Zero

```python
class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        f = [0] * (n + 1)
        pre = 0
        for i, v in enumerate(nums):
            pre += f[i]
            v += pre 
            if v == 0:
                continue
            if v < 0 or i + k > n:
                return False
            pre += -v
            f[i + k] -= -v 
        return True
```


### 2528. Maximize the Minimum Powered City

```python
class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        n = len(stations)
        pre = list(accumulate(stations, initial=0))
        for i in range(n):
            stations[i] = pre[min(i + r + 1, n)] - pre[max(i - r, 0)]

        def check(min_power: int) -> bool:
            f = [0] * n 
            sum_f = need = 0
            for i, power in enumerate(stations):
                sum_f += f[i] 
                m = min_power - power - sum_f
                if m > 0: 
                    need += m
                    if need > k: 
                        return False
                    sum_f += m 
                    if i + r * 2 + 1 < n: 
                        f[i + r * 2 + 1] -= m
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

### 1589. Maximum Sum Obtained of Any Permutation

```python
class Solution:
    def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        n = len(nums)
        f = [0] * (n + 1)
        for s, e in requests:
            f[s] += 1
            f[e + 1] -= 1
        pre = list(accumulate(f))[:-1]
        pre.sort()
        nums.sort()
        res = sum(a * b for a, b in zip(pre, nums))
        return res % mod
```

### 2015. Average Height of Buildings in Each Segment

```python
class Solution:
    def averageHeightOfBuildings(self, buildings: List[List[int]]) -> List[List[int]]:
        d = defaultdict(lambda: [0, 0])
        for s, e, h in buildings:
            d[s][0] += 1
            d[s][1] += h 
            d[e][0] -= 1
            d[e][1] -= h 
        arr = sorted([[pos, cnt, val] for pos, (cnt, val) in d.items()])
        for i in range(1, len(arr)):
            arr[i][1] += arr[i - 1][1]
            arr[i][2] += arr[i - 1][2]
        res = []
        for a, b in pairwise(arr):
            k1, c1, v1 = a 
            k2, c2, v2 = b 
            if v1:
                res.append([k1, k2, v1 // c1])
        q = deque(res)
        ans = []
        while len(q) > 1:
            s1, e1, v1 = q.popleft()
            s2, e2, v2 = q.popleft()
            if s2 == e1 and v1 == v2:
                q.appendleft([s1, e2, v1])
            else:
                ans.append([s1, e1, v1])
                q.appendleft([s2, e2, v2])
        if q:
            ans.append(q[0])
        return ans
```

### 2536. Increment Submatrices by One

```python
class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        # dp = [[0] * n for r in range(n)]
        # for x1, y1, x2, y2 in queries:
        #     for r in range(x1, x2 + 1):
        #         for c in range(y1, y2 + 1):
        #             dp[r][c] += 1
        # return dp

        # f = [[0] * (n + 1) for r in range(n)]
        # for x1, y1, x2, y2 in queries:
        #     for r in range(x1, x2 + 1):
        #         f[r][y1] += 1
        #         f[r][y2 + 1] -= 1
        # return [list(accumulate(row[:-1])) for row in f]

        f = [[0] * (n + 2) for r in range(n + 2)]
        for x1, y1, x2, y2 in queries:
            f[x1 + 1][y1 + 1] += 1
            f[x1 + 1][y2 + 2] -= 1
            f[x2 + 2][y1 + 1] -= 1
            f[x2 + 2][y2 + 2] += 1
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                f[r][c] += f[r][c - 1] + f[r - 1][c] - f[r - 1][c - 1]
        return [row[1:-1] for row in f[1:-1]]
```

### 850. Rectangle Area II

- https://leetcode.cn/problems/rectangle-area-ii/solutions/1826992/gong-shui-san-xie-by-ac_oier-9r36/

```python
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        def getArea(width):
            lines = [(y1, y2) for x1, y1, x2, y2 in rectangles if x1 <= a and x2 >= b]
            lines.sort()
            height, start, end = 0, -1, -1
            for s, e in lines:
                if s > end:
                    height += end - start
                    start = s
                end = max(end, e)
            height += end - start
            return width * height

        mod = 10 ** 9 + 7
        points = []
        for x1, y1, x2, y2 in rectangles:
            points.extend([x1, x2])
        points.sort()

        res = 0
        for i in range(1, len(points)):
            a, b = points[i - 1], points[i]
            width = b - a 
            area = getArea(width)
            res += area
        return res % mod
```

### 2132. Stamping the Grid

```python
class Solution:
    def possibleToStamp(self, grid: List[List[int]], stampHeight: int, stampWidth: int) -> bool:
        R, C = len(grid), len(grid[0])
        pre = [[0] * (C + 1) for r in range(R + 1)]
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                pre[r][c] = pre[r - 1][c] + pre[r][c - 1] - pre[r - 1][c - 1] + grid[r - 1][c - 1]

        f = [[0] * (C + 2) for r in range(R + 2)]
        for x2 in range(stampHeight, R + 1):
            for y2 in range(stampWidth, C + 1):
                x1, y1 = x2 - stampHeight + 1, y2 - stampWidth + 1
                if pre[x2][y2] - pre[x2][y1 - 1] - pre[x1 - 1][y2] + pre[x1 - 1][y1 - 1] == 0:
                    f[x1][y1] += 1
                    f[x1][y2 + 1] -= 1
                    f[x2 + 1][y1] -= 1
                    f[x2 + 1][y2 + 1] += 1
                    
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                f[r][c] += f[r][c - 1] + f[r - 1][c] - f[r - 1][c - 1]
                if grid[r - 1][c - 1] == 0 and f[r][c] == 0:
                    return False
        return True
```