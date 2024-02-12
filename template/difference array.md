## difference array

- 1d difference array

* [370. Range Addition](#370-Range-Addition)
* [1109. Corporate Flight Bookings](#1109-Corporate-Flight-Bookings)
* [1893. Check if All the Integers in a Range Are Covered](#1893-Check-if-All-the-Integers-in-a-Range-Are-Covered)
* [2848. Points That Intersect With Cars](#2848-Points-That-Intersect-With-Cars)
* [848. Shifting Letters](#848-Shifting-Letters)
* [2381. Shifting Letters II](#2381-Shifting-Letters-II)
* [2406. Divide Intervals Into Minimum Number of Groups](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups)
* [2772. Apply Operations to Make All Array Elements Equal to Zero](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups)
* [2237. Count Positions on Street With Required Brightness](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups)
* [1589. Maximum Sum Obtained of Any Permutation](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups)
* [1943. Describe the Painting](#2406-Divide-Intervals-Into-Minimum-Number-of-Groups)

- 2d difference array

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

### 1109. Corporate Flight Bookings

```python
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        f = [0] * (n + 2)
        for first, last, seats in bookings:
            f[first] += seats
            f[last + 1] -= seats
        prefix = list(accumulate(f))
        return prefix[1:-1]
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

### 848. Shifting Letters

```python
class Solution:
    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        prefix = list(accumulate(shifts[::-1]))[::-1]
        res = ''
        for c, n in zip(s, prefix):
            res += chr((n + ord(c) - 97) % 26 + 97)
        return res
```

### 2381. Shifting Letters II

```python
class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        f = [0] * (len(s) + 1)
        for start, end, sign in shifts:
            if sign == 1:
                f[start] += 1
                f[end + 1] -= 1
            else:
                f[start] += -1
                f[end + 1] -= -1
        res = ''
        for i, n in enumerate(list(accumulate(f))[:-1]):
            res += chr((n + ord(s[i]) - ord('a')) % 26 + ord('a'))
        return res
```

### 2406. Divide Intervals Into Minimum Number of Groups

```python
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        f = [0] * (10 ** 6 + 2)
        for l, r in intervals:
            f[l] += 1
            f[r + 1] -= 1
        f = list(accumulate(f))
        return max(f)
```

### 2772. Apply Operations to Make All Array Elements Equal to Zero

```python
class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        f = [0] * (n + 1)
        sum_f = 0
        for i, v in enumerate(nums):
            sum_f += f[i]
            v += sum_f
            if v == 0:
                continue
            if v < 0 or i + k > n:
                return False
            sum_f -= v
            f[i + k] += v
        return True
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

### 1943. Describe the Painting

```python
class Solution:
    def splitPainting(self, segments: List[List[int]]) -> List[List[int]]:
        color = defaultdict(int)
        for s, e, c in segments:
            color[s] += c
            color[e] -= c 
        points = sorted([[k, v]for k, v in color.items()])

        n = len(points)
        for i in range(1, n):
            points[i][1] += points[i-1][1]

        res = []
        for i in range(n - 1):
            if points[i][1]:
                res.append([points[i][0], points[i + 1][0], points[i][1]])
        return res
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
            if (A[i] + flip) % 2 == 0: #需要翻转
                if i + K > n: #出界了，就结束
                    return -1
                ans += 1 # 翻转次数
                flip += 1 # 左侧位置+1 直接传递到 revCnt 上
                f[i + K] -= 1 # 右端点+1 位置 -1
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