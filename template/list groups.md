## template 1: start any position

### 228, 1446, 1578, 1759, 1839, 1869, 1957, 2038, 2110

```python
i, n = 0, len(nums)
while i < n:
    start = i
    while i < n - 1 and ...:
        i += 1
```

## template 2: start with condition

### 2760, 2765

```python
i, n = 0, len(nums)
while i < n or i < n - 1:
    if ...:
        i += 1
        continue
    start = i
    while i < n - 1 and ...:
        i += 1
```

### list groups

- 11 similar questions

### 228. Summary Ranges
### 1446. Consecutive Characters
### 1578. Minimum Time to Make Rope Colorful
### 1759. Count Number of Homogenous Substrings
### 1839. Longest Substring Of All Vowels in Order
### 1869. Longer Contiguous Segments of Ones than Zeros
### 1957. Delete Characters to Make Fancy String
### 2038. Remove Colored Pieces if Both Neighbors are the Same Color
### 2110. Number of Smooth Descent Periods of a Stock
### 2760. Longest Even Odd Subarray With Threshold
### 2765. Longest Alternating Subarray
### 186. Reverse Words in a String II
### 2943. Maximize Area of Square Hole in Grid
### 2981. Find Longest Special Substring That Occurs Thrice I
### 2982. Find Longest Special Substring That Occurs Thrice II


### 228. Summary Ranges

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        res, i, n = [], 0, len(nums)
        while i < n:
            start = i
            while i < n - 1 and nums[i] + 1 == nums[i + 1]:
                i += 1
            s = str(nums[start])
            if start < i:
                s += '->' + str(nums[i])
            res.append(s)
            i += 1
        return res
```

### 1446. Consecutive Characters

```python
class Solution:
    def maxPower(self, s: str) -> int:
        res, i, n = 1, 0, len(s)
        while i < n:
            start = i
            while i < n - 1 and s[i] == s[i + 1]:
                i += 1
            res = max(res, i - start + 1)
            i += 1
        return res
```

### 1578. Minimum Time to Make Rope Colorful

```python
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        res, i, n = 0, 0, len(colors)
        while i < n:
            start = i
            while i < n - 1 and colors[i] == colors[i + 1]:
                i += 1
            res += sum(neededTime[start: i + 1]) - max(neededTime[start: i + 1])
            i += 1
        return res 
```

### 1759. Count Number of Homogenous Substrings

```python
class Solution:
    def countHomogenous(self, s: str) -> int:
        res, i, n, mod = 0, 0, len(s), 10 ** 9 + 7
        while i < n:
            start = i
            while i < n - 1 and s[i] == s[i + 1]:
                i += 1
            c = i - start + 1
            res += (c * (c + 1)) // 2
            i += 1
        return res % mod
```

### 1839. Longest Substring Of All Vowels in Order

```python
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:
        res, i, n = 0, 0, len(word)
        while i < n:
            start = i
            while i < n - 1 and word[i] <= word[i + 1]:
                i += 1
            if len(set(list(word[start: i + 1]))) == 5:
                res = max(res, i - start + 1)
            i += 1
        return res
```

### 1869. Longer Contiguous Segments of Ones than Zeros

```python
class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        ones, zeros, i, n = 0, 0, 0, len(s)
        while i < n:
            start = i
            while i < n - 1 and s[i] == s[i + 1]:
                i += 1
            if s[start] == '1':
                ones = max(ones, i - start + 1)
            else:
                zeros = max(zeros, i - start + 1)
            i += 1
        return ones > zeros
```

### 1957. Delete Characters to Make Fancy String

```python
class Solution:
    def makeFancyString(self, s: str) -> str:
        res, i, n = '', 0, len(s)
        while i < n:
            start = i
            while i < n - 1 and s[i] == s[i + 1]:
                i += 1
            ans = s[start: min(start + 2, i + 1)]
            res += ans
            i += 1
        return res
```

### 2038. Remove Colored Pieces if Both Neighbors are the Same Color

```python
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        A, B, i, n = 0, 0, 0, len(colors)
        while i < n:
            start = i
            while i < n - 1 and colors[i] == colors[i + 1]:
                i += 1
            if i - start + 1 >= 3:
                if colors[start] == 'A':
                    A += i - start + 1 - 2
                else:
                    B += i - start + 1 - 2
            i += 1
        return A > B
```

### 2110. Number of Smooth Descent Periods of a Stock

```python
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        res, i, n = 0, 0, len(prices)
        while i < n:
            start = i
            while i < n - 1 and prices[i] - 1 == prices[i + 1]:
                i += 1
            c = i - start + 1
            res += (c * (c + 1)) // 2
            i += 1
        return res
```

### 2760. Longest Even Odd Subarray With Threshold

```python
class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        res, i, n = 0, 0, len(nums)
        while i < n:
            if nums[i] % 2 or nums[i] > threshold:
                i += 1
                continue
            start = i
            while i < n - 1 and nums[i] % 2 != nums[i + 1] % 2 and nums[i + 1] <= threshold:
                i += 1
            res = max(res, i - start + 1)
            i += 1
        return res
```

### 2765. Longest Alternating Subarray

```python
class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        res, i, n = -1, 0, len(nums)
        while i < n - 1:
            if nums[i + 1] != nums[i] + 1:
                i += 1
                continue
            start = i
            while i < n - 1 and nums[i + 1] == nums[start] + (i - start + 1) % 2:
                i += 1
            res = max(res, i - start + 1)
        return res
```

### 2981. Find Longest Special Substring That Occurs Thrice I

```python
class Solution:
    def maximumLength(self, s: str) -> int:
        def check(s):
            n = len(s)
            letter = s[0]
            for i in range(n):
                c[letter * (i + 1)] += n - i 
                
        n, c, i = len(s), Counter(), 0
        while i < n:
            start = i
            j = start 
            while j < n and s[j] == s[start]:
                j += 1
            check(s[start: j])
            i = j
        res = [len(k) for k, v in c.items() if v >= 3]
        return max(res) if res else -1
```

### 2982. Find Longest Special Substring That Occurs Thrice II

```python
class Solution:
    def maximumLength(self, s: str) -> int:
        def check(s):
            n = len(s)
            letter = s[0]
            for i in range(n - 1, -1, -1):
                c[letter * (i + 1)] += n - i 
                if c[letter * (i + 1)] >= 3:
                    break
                
        n, c, i = len(s), Counter(), 0
        while i < n:
            start = i
            j = start 
            while j < n and s[j] == s[start]:
                j += 1
            check(s[start: j])
            i = j
        res = [len(k) for k, v in c.items() if v >= 3]
        return max(res) if res else -1
```

### 2943. Maximize Area of Square Hole in Grid

```python
class Solution:
    def maximizeSquareHoleArea(self, n: int, m: int, hBars: List[int], vBars: List[int]) -> int:
        hBars.sort()
        vBars.sort()
        def check(nums):
            res, i, n = 0, 0, len(nums)
            while i < n:
                start = i
                while i < n - 1 and nums[i] + 1 == nums[i + 1]:
                    i += 1
                res = max(res, i - start + 1)
                i += 1
            return res
        h, v = check(hBars) + 1, check(vBars) + 1
        return min(h, v) ** 2
```

### Find if Array Can Be Sorted

```python
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        i = 0
        n, res = len(nums), []
        while i < n:
            start = i 
            j = start 
            while j < n and nums[j].bit_count() == nums[start].bit_count():
                j += 1
            res.append(sorted(nums[start: j]))
            i = j
        return sum(res, []) == sorted(nums)
```

### 186. Reverse Words in a String II

```python
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def subReverse(l, r):
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
    
        s.reverse()
        i = 0
        while i < len(s):
            if s[i] != ' ':
                start = i
                j = start 
                while j < len(s) and s[j] != ' ':
                    j += 1
                subReverse(start, j - 1)
                i = j 
            else:
                i += 1
        return s
```

### 1784. Check if Binary String Has at Most One Segment of Ones

```python
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        count = 0
        i = 0
        while i < len(s):
            if s[i] == '1':
                start = i
                j = start
                while j < len(s) and s[j] == '1':
                    j += 1
                count += 1
                i = j 
            i += 1
        return count <= 1
```

### 551. Student Attendance Record I

```python
class Solution:
    def checkRecord(self, s: str) -> bool:
        i = 0
        late, absent = 0, s.count('A')
        while i < len(s):
            start = i
            j = start
            while j < len(s) and s[j] == s[start]:
                j += 1
            if s[start] == 'L':
                late = max(late, j - start)
            i = j 
        return absent < 2 and late < 3
```