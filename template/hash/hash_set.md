## Hash Set

### 1461. Check If a String Contains All Binary Codes of Size K

```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        if len(s) < k:
            return False

        s_set = set()
        for i in range(len(s) - k + 1):
            s_set.add(s[i: i + k])
        return len(s_set) == 2 ** k
```

### 336. Palindrome Pairs

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        lookup = {w: i for i, w in enumerate(words)}
        res = []
        for i, w in enumerate(words):
            for j in range(len(w) + 1):
                pre, suf = w[:j], w[j:]
                if pre[::-1] == pre and suf[::-1] != w and suf[::-1] in lookup:
                    res.append([lookup[suf[::-1]], i])
                if suf[::-1] == suf and pre[::-1] != w and pre[::-1] in lookup and j != len(w):
                    # j != len(w)，j = w的情况已经出现过, avoid duplicate
                    res.append([i, lookup[pre[::-1]]])
        return res
```

### 594. Longest Harmonious Subsequence

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        d = Counter(nums)
        res = 0
        for k in d:
            if k - 1 in d:
                res = max(res, d[k] + d[k - 1])
            if k + 1 in d:
                res = max(res, d[k] + d[k + 1])
        return res
```

### 1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers

```python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        def check(nums1, nums2):
            c1, c2 = Counter(nums1), Counter(nums2)
            res = 0
            for n1 in c1:
                for n2 in c2:
                    if (n1 * n1) % n2 == 0:
                        if n1 * n1 // n2 in c2 and n1 == n2:
                            res += c1[n1] * c2[n2] * (c2[n2] - 1)
                        if n1 * n1 // n2 in c2 and n1 != n2:
                            res += c1[n1] * c2[n2] * c2[n1 * n1 // n2]
            return res // 2
        return check(nums1, nums2) + check(nums2, nums1)
```

### 1647. Minimum Deletions to Make Character Frequencies Unique

```python
class Solution:
    def minDeletions(self, s: str) -> int:
        c = Counter(s)
        seen = set()
        res = 0
        for v in sorted(c.values()):
            if v not in seen:
                seen.add(v)
            else:
                while v in seen and v - 1 >= 0:
                    v -= 1
                    res += 1
                seen.add(v)
        return res
```

### 890. Find and Replace Pattern

```python
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        def check(w):
            c1, c2 = Counter(), Counter()
            for a, b in zip(w, pattern):
                c1[a] = b 
                c2[b] = a 
            res1, res2 = '', ''
            for c in w:
                res1 += c1[c]
            for c in pattern:
                res2 += c2[c]
            return res1 == pattern and res2 == w
        res = []
        for w in words:
            if check(w):
                res.append(w)
        return res
```

### 2225. Find Players With Zero or One Losses

```python
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        c_lose = Counter()
        s = set()
        for a, b in matches:
            c_lose[b] += 1
            s.add(a)
            s.add(b)
        res1, res2 = [], []
        for i in s:
            if i not in c_lose:
                res1.append(i)
            elif c_lose[i] == 1:
                res2.append(i)
        return [sorted(res1), sorted(res2)]
```

### 2244. Minimum Rounds to Complete All Tasks

```python
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        c = Counter(tasks)
        res = 0
        for v in c.values():
            if v % 3 == 0:
                res += v // 3
            else:
                if v > 3:
                    res += v // 3 + 1
                else:
                    if v == 1:
                        return -1
                    else:
                        res += 1
        return res
```

### 2482. Difference Between Ones and Zeros in Row and Column

```python
class Solution:
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        rows, cols = defaultdict(int), defaultdict(int)
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    rows[r] += 1
                    cols[c] += 1
        dp = [[0] * C for r in range(R)]
        for r in range(R):
            for c in range(C):
                dp[r][c] = rows[r] + cols[c] - (R + C - (rows[r] + cols[c]))
        return dp
```

### 1224. Maximum Equal Frequency

```python
class Solution:
    def maxEqualFreq(self, nums: List[int]) -> int:
        n = len(nums)
        count = Counter(nums)
        countFreq = Counter(count.values())
        for i in range(n - 1, -1, -1):
            if len(countFreq) == 1:
                freq, num = list(countFreq.items())[0]
                if freq == 1 or num == 1:
                    return i + 1
            elif len(countFreq) == 2:
                lst = list(sorted(countFreq.items()))  
                freq1, num1 = lst[0]
                freq2,num2 = lst[1]
                if (freq1 == 1 and num1 == 1) or (num2 == 1 and freq2 == freq1 + 1):
                    return i + 1
            f = count[nums[i]]
            count[nums[i]] -= 1
            if count[nums[i]] == 0:
                del count[nums[i]]
            countFreq[f] -= 1
            if countFreq[f] == 0:
                del countFreq[f]
            if f - 1 > 0:
                countFreq[f - 1] += 1
        return 2
```

### 2593. Find Score of an Array After Marking All Elements

```python
class Solution:
    def findScore(self, nums: List[int]) -> int:
        a = sorted([(i, n) for i, n in enumerate(nums)], key = lambda x: x[1])
        res = 0
        s = set()
        for i, n in a:
            if i not in s:
                res += n 
                s.add(i)
                s.add(i - 1)
                s.add(i + 1)
        return res 
```

### 2061. Number of Spaces Cleaning Robot Cleaned

```python
class Solution:
    def numberOfCleanRooms(self, room: List[List[int]]) -> int:
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        R, C = len(room), len(room[0])
        visited = set()
        r, c, d = 0, 0, 0
        while (r, c, d) not in visited:
            visited.add((r, c, d))
            dr, dc = directions[d]
            row, col = r + dr, c + dc 
            if 0 <= row < R and 0 <= col < C and room[row][col] == 0:
                r, c = row, col 
            else:
                d = (d + 1) % 4
        return len(set((r, c) for r, c, d in visited))
```

### 2354. Number of Excellent Pairs

```python
class Solution:
    def countExcellentPairs(self, nums: List[int], k: int) -> int:
        d = Counter([n.bit_count() for n in set(nums)])
        res = 0
        for k1, v1 in d.items():
            for k2, v2 in d.items():
                if k1 + k2 >= k:
                    res += v1 * v2 
        return res
```