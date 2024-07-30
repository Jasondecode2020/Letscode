# sweep line(13)

## 1d sweep line(12)

* 1. [252. Meeting Rooms](#252-meeting-rooms) 1300
* 2. [253. Meeting Rooms II](#253-meeting-rooms-ii) 1400
* 3. [56. Merge Intervals](#56-merge-intervals) 1500
* 4. [57. Insert Interval](#57-insert-interval) 1550
* 5. [1854. Maximum Population Year](#1854-maximum-population-year) 1600
* 6. [2021. Brightest Position on Street](#2021-brightest-position-on-street) 1600
* 7. [759. Employee Free Time](#759-employee-free-time) 1700
* 8. [729. My Calendar I](#729-my-calendar-i) 1800
* 9. [731. My Calendar II](#731-my-calendar-ii) 1800
* 10. [732. My Calendar III](#732-my-calendar-iii) 1800
* 11. [3169. Count Days Without Meetings](#3169-count-days-without-meetings) 1700
* 12. [ 2963. Count the Number of Good Partitions](#2963-count-the-number-of-good-partitions) 1984

## 2d sweep line(1)

* [218. The Skyline Problem](#218-the-skyline-problem) 2500

### 252. Meeting Rooms

- use count to check overlap

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

- difference array

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

### 253. Meeting Rooms II

- use count to store numbers of overlap

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

### 56. Merge Intervals

- use count to store one result
- use start, end to record one merged interval

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        res, count = [], 0
        start, end = inf, -inf
        for point, sign in events:
            if sign < 0:
                start = min(start, point)
                count += 1
            else:
                end = max(end, point)
                count -= 1
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf
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

### 1854. Maximum Population Year

- use maxCount to record start year

```python
class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        events = []
        for s, e in logs:
            events.append([s, 1])
            events.append([e, -1])
        events.sort()

        res, count, maxCount = 0, 0, 0
        for year, sign in events:
            if sign == 1:
                count += 1
                if count > maxCount:
                    res = year
                    maxCount = count  
            else:
                count -= 1
        return res
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


### 729. My Calendar I

```python
from sortedcontainers import SortedList
class MyCalendar:

    def __init__(self):
        self.events = SortedList()

    def book(self, start: int, end: int) -> bool:
        self.events.add((start, 1))
        self.events.add((end, -1))
        count = 0
        for point, sign in self.events:
            count += sign
            if count > 1:
                self.events.remove((start, 1))
                self.events.remove((end, -1))
                return False
        return True
```

### 731. My Calendar II

```python
class MyCalendarTwo:

    def __init__(self):
        self.intervals = []

    def book(self, start: int, end: int) -> bool:
        self.intervals.append([start, end])
        events = []
        for s, e in self.intervals:
            events.append((s, 1))
            events.append((e, -1))
        events.sort()
        count = 0
        for point, sign in events:
            count += sign
            if count >= 3:
                self.intervals.pop()
                return False
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

### 3169. Count Days Without Meetings

```python
class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        intervals = meetings[::]
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        res, count = [], 0
        start, end = inf, -inf
        for point, sign in events:
            if sign < 0:
                start = min(start, point)
                count += 1
            else:
                end = max(end, point)
                count -= 1
            if count == 0:
                res.append([start, end])
                start, end = inf, -inf
        working_days = 0
        for a, b in res:
            working_days += b - a + 1
        return days - working_days
```

###  2963. Count the Number of Good Partitions

```python
class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10 ** 9 + 7
        d = defaultdict(int)
        for i, n in enumerate(nums):
            if n in d:
                d[n][1] = i 
            else:
                d[n] = [i, i]
        
        a = sorted(d.values())
        res = 0
        r = a[0][1]
        for left, right in a[1:]:
            if left > r:
                res += 1
            r = max(r, right)
        return pow(2, res, mod)
```

### 218. The Skyline Problem

- T: O(n^2)

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        points, res, max_heap = [], [], [0]
        for left, right, height in buildings:
            points.append([left, height, 1])
            points.append([right, -height, -1])
        points.sort(key = lambda x: (x[0], -x[1]))
        for x, height, state in points:
            if state == 1:
                if height > -max_heap[0]:
                    res.append([x, height])
                heappush(max_heap, -height)
            else:
                max_heap.remove(height)
                heapify(max_heap)
                if -height > -max_heap[0]:
                    res.append([x, -max_heap[0]])
        return res
```

- T: O(nlog(n))

```python
from sortedcontainers import SortedList
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        points, res, sl = [], [], SortedList([0])
        for left, right, height in buildings:
            points.append([left, height, 1])
            points.append([right, -height, -1])
        points.sort(key = lambda x: (x[0], -x[1]))
        for x, height, state in points:
            if state == 1:
                if height > sl[-1]:
                    res.append([x, height])
                sl.add(height)
            else:
                sl.remove(-height)
                if -height > sl[-1]:
                    res.append([x, sl[-1]])
        return res
```