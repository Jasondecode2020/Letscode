## difference array

- 1d difference array

```html
<p>Can calculate directly using difference array</p>
```
* [2848. Points That Intersect With Cars](#2848-Points-That-Intersect-With-Cars) 1230
* [1893. Check if All the Integers in a Range Are Covered](#1893-Check-if-All-the-Integers-in-a-Range-Are-Covered) 1307
* [1854. Maximum Population Year](#1854-maximum-population-year) 1370
* [1094. Car Pooling](#1094-car-pooling) 1441
* [1109. Corporate Flight Bookings](#1109-Corporate-Flight-Bookings) 1570

* [56. Merge Intervals](#56-merge-intervals) 1580
* [57. Insert Interval](#57-insert-interval) 1600
* [732. My Calendar III](#732-my-calendar-iii) 1700
* [2406. Divide Intervals Into Minimum Number of Groups](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups) 1731

* [2381. Shifting Letters II](#2381-Shifting-Letters-II) 1793
* [995. Minimum Number of K Consecutive Bit Flips](#995-minimum-number-of-k-consecutive-bit-flips) 1835
* [1589. Maximum Sum Obtained of Any Permutation](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups) 1871
* [1943. Describe the Painting](#1943-describe-the-painting) 1969
* [2251. Number of Flowers in Full Bloom](#2251-number-of-flowers-in-full-bloom) 2251

* [2772. Apply Operations to Make All Array Elements Equal to Zero](#2772-apply-operations-to-make-all-array-elements-equal-to-zero) 2029
* [798. Smallest Rotation with Highest Score](#798-smallest-rotation-with-highest-score) 2129
* [2528. Maximize the Minimum Powered City](#2528-maximize-the-minimum-powered-city) 2236
* [1674. Minimum Moves to Make Array Complementary](#1674-minimum-moves-to-make-array-complementary) 2333
* [3017. Count the Number of Houses at a Certain Distance II]() 2709

* [253. Meeting Rooms II](#253-meeting-rooms-ii)
* [370. Range Addition](#370-Range-Addition)
* [759. Employee Free Time](#759-employee-free-time)
* [2021. Brightest Position on Street](#2021-brightest-position-on-street)
* [2015. Average Height of Buildings in Each Segment](#2015-average-height-of-buildings-in-each-segment)

* [2237. Count Positions on Street With Required Brightness](#2237-count-positions-on-street-with-required-brightness)
* [3009. Maximum Number of Intersections on the Chart](#3009-maximum-number-of-intersections-on-the-chart) 2500

- 2d difference array (3)

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

```c++
class Solution {
public:
    int numberOfPoints(vector<vector<int>>& nums) {
        int f[102]{};
        for (auto &p: nums) {
            f[p[0]]++;
            f[p[1] + 1]--;
        }
        int res = 0, pre = 0;
        for (int n: f) {
            pre += n;
            res += pre > 0;
        }
        return res;
    }
};
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

### 1854. Maximum Population Year

```python
class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        f = [0] * 2051
        for s, e in logs:
            f[s] += 1
            f[e] -= 1
        pre = list(accumulate(f))
        ans, res = 0, 0
        for i, n in enumerate(pre):
            if n > ans:
                res = i 
                ans = n 
        return res 
```

### 2960. Count Tested Devices After Test Operations

```python
class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        dec = 0
        for x in batteryPercentages:
            if x > dec:
                dec += 1
        return dec
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

```c++
class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        int f[1001]{};
        for (auto &p: trips) {
            f[p[1]] += p[0];
            f[p[2]] -= p[0];
        }
        int pre = 0;
        for (int n: f) {
            pre += n;
            if (pre > capacity) {
                return false;
            }
        }
        return true;
    }
};
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

### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        n = len(intervals)
        intervals.sort()
        start, end = intervals[0]
        res = []
        for i in range(1, n):
            s, e = intervals[i]
            if s > end:
                res.append([start, end])
                start = s 
            end = max(end, e)
        res.append([start, end])
        return res 
```

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        events = []
        for s, e in intervals:
            events.append([s, -1])
            events.append([e, 1])
        events.sort()
        start, end = inf, -inf
        res = []
        count = 0
        for x, sign in events:
            if sign == -1:
                count += 1
                start = min(start, x)
            else:
                count -= 1
                end = max(end, x)
            if count == 0:
                res.append([start, end])
                start = inf 
        return res
```

### 57. Insert Interval

- use count to store one result
- use start, end to record one merged interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        count, res = 0, []
        start, end = inf, -inf
        for point, sign in events:
            if sign < 0:
                count += 1
                start = min(start, point)
            else:
                count -= 1
                end = max(end, point)
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf
        return res
```

- better O(n) solution

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        s, e = newInterval 
        res = []
        insert = False 
        for start, end in intervals:
            if e < start:
                if not insert:
                    res.append([s, e])
                    insert = True
                res.append([start, end])
            elif end < s:
                res.append([start, end])
            else:
                s, e = min(s, start), max(e, end)
        if not insert:
            res.append([s, e])
        return res 
```


### 732. My Calendar III

```python
class MyCalendarThree:

    def __init__(self):
        self.res = -inf
        self.intervals = []

    def book(self, start: int, end: int) -> int:
        self.intervals.append([start, end])
        events = []
        for s, e in self.intervals:
            events.append((s, 1))
            events.append((e, -1))
        events.sort()
        count = 0
        for point, sign in events:
            count += sign
            self.res = max(self.res, count)
        return self.res
```

```python
from sortedcontainers import SortedList
class MyCalendarThree:

    def __init__(self):
        self.events = SortedList()

    def book(self, startTime: int, endTime: int) -> int:
        self.events.add((startTime, 1))
        self.events.add((endTime, -1))
        res, count = 0, 0
        for x, sign in self.events:
            if sign == 1:
                count += 1
                res = max(res, count)
            else:
                count -= 1
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

### 1589. Maximum Sum Obtained of Any Permutation

```python
class Solution:
    def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        nums.sort(reverse = True)
        n = len(nums)
        f = [0] * (n + 1)
        for s, e in requests:
            f[s] += 1
            f[e + 1] -= 1
        arr = list(accumulate(f[:-1]))
        arr.sort(reverse = True)
        return sum(a * b for a, b in zip(arr, nums)) % mod
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

### 798. Smallest Rotation with Highest Score

```python
class Solution:
    def bestRotation(self, nums: List[int]) -> int:
        n = len(nums)
        diff = [0] * (n + 1)
        for i, num in enumerate(nums):
            if i >= num:
                diff[0] += 1
                diff[i - num + 1] -= 1
                diff[i + 1] += 1
            else:
                diff[i + 1] += 1
                diff[i + n + 1 - num] -= 1
        pre = list(accumulate(diff))
        mx = max(pre)
        for i, n in enumerate(pre):
            if n == mx:
                return i
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

### 253. Meeting Rooms II

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        n = max(b for a, b in intervals)
        f = [0] * (n + 2)
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

### 995. Minimum Number of K Consecutive Bit Flips

```python
class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        n = len(A)
        f = [0] * (n + 1)
        ans, flip = 0, 0
        for i in range(n):
            flip += f[i]
            if (A[i] + flip) % 2 == 0: #需要翻转 1, 0
                if i + K > n: #出界了，就结束
                    return -1
                ans += 1 # 翻转次数
                flip += 1 # 左侧位置+1 直接传递到 revCnt 上
                f[i + K] -= 1 # 右端点+1 位置 -1
        return ans
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

### 759. Employee Free Time

'''
We are given a list schedule of employees, which represents the working time for each employee.

Each employee has a list of non-overlapping Intervals, and these intervals are in sorted order.

Return the list of finite intervals representing common, positive-length free time for all employees, also in sorted order.

(Even though we are representing Intervals in the form [x, y], the objects inside are Intervals, not lists or arrays. For example, schedule[0][0].start = 1, schedule[0][0].end = 2, and schedule[0][0][0] is not defined).  Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length.

Example 1:

Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation: There are a total of three employees, and all common
free time intervals would be [-inf, 1], [3, 4], [10, inf].
We discard any intervals that contain inf as they aren't finite.
Example 2:

Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]

Constraints:

1 <= schedule.length , schedule[i].length <= 50
0 <= schedule[i].start < schedule[i].end <= 10^8
'''

```python
class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        events = []
        for s in schedule:
            for item in s:
                events.append([item.start, -1])
                events.append([item.end, 1])
        events.sort()

        count, res = 0, []
        start, end = inf, -inf
        for point, sign in events:
            if sign < 0:
                count += 1
                start = min(start, point)
            else:
                count -= 1
                end = max(end, point)
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf
        return [Interval(res[i - 1][1], res[i][0]) for i in range(1, len(res))]
```


### 2021. Brightest Position on Street

```python
class Solution:
    def brightestPosition(self, lights: List[List[int]]) -> int:
        events = []
        for point, radius in lights:
            s, e = point - radius, point + radius
            events.append((s, -1))
            events.append((e, 1))
        events.sort()
        res, count, ans = -inf, 0, 0
        for x, sign in events:
            if sign == -1:
                count += 1
            else:
                count -= 1
            if count > ans:
                res = x 
                ans = count
        return res 
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

### 830. Positions of Large Groups

```python
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        i, res = 0, []
        while i < len(s):
            start = i 
            j = start
            while j < len(s) and s[j] == s[start]:
                j += 1
            if j - start >= 3:
                res.append([start, j - 1])
            i = j
        return res
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