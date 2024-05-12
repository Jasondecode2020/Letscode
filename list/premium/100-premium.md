## Merge intervals (2)

* [758. Bold Words in String](#758-bold-words-in-string)
* [616. Add Bold Tag in String](#616-add-bold-tag-in-string)

## Graph BFS (1)

* [1197. Minimum Knight Moves](#1197-minimum-knight-moves)


### 616. Add Bold Tag in String
### 758. Bold Words in String (same as 616)

```python
class Solution:
    def addBoldTag(self, s: str, words: List[str]) -> str:
        n = len(s)
        intervals = [] 
        words = set(words)
        for i in range(n):
            for w in words:
                if s[i:].startswith(w):
                    if not intervals or intervals[-1][1] < i:
                        intervals.append([i, i + len(w)])
                    else:
                        intervals[-1][1] = max(intervals[-1][1], i + len(w))

        res = ''
        prev = 0
        if not intervals:
            return s
        for a, b in intervals:
            res += s[prev: a] + '<b>' + s[a: b] + '</b>'
            prev = b 
        return res + s[b:]
```
## BFS 

* [1197. Minimum Knight Moves](#1197-minimum-knight-moves)

### 1197. Minimum Knight Moves

```python
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        directions = [[2, 1], [1, 2], [-1, 2], [-2, 1], [-2, -1], [-1, -2], [1, -2], [2, -1]]
        q = deque([(0, 0, 0)])
        visited = set([(0, 0)])
        while q:
            r, c, step = q.popleft()
            if r == x and c == y:
                return step
            for dr, dc in directions:
                row, col = r + dr, c + dc 
                if (row, col) not in visited:
                    visited.add((row, col))
                    q.append((row, col, step + 1))
```

### 254. Factor Combinations

```python
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        def backtrack(n, ans):
            if len(ans) > 0:
                res.append(ans + [n])
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    if not ans or i >= ans[-1]:
                        backtrack(n // i, ans + [i])

        res = []
        backtrack(n, [])
        return res 
```

### sliding window (4)

* [159. Longest Substring with At Most Two Distinct Characters](#159-longest-substring-with-at-most-two-distinct-characters)
* [340. Longest Substring with At Most K Distinct Characters](#340-longest-substring-with-at-most-k-distinct-characters)
* [487. Max Consecutive Ones II](#487-max-consecutive-ones-ii)
* [1100. Find K-Length Substrings With No Repeated Characters](#1100-find-k-length-substrings-with-no-repeated-characters)

### 159. Longest Substring with At Most Two Distinct Characters

```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        d = Counter()
        l, res = 0, 0
        for r, c in enumerate(s):
            d[c] += 1
            while len(d) > 2:
                d[s[l]] -= 1
                if d[s[l]] == 0:
                    d.pop(s[l])
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 340. Longest Substring with At Most K Distinct Characters

```python
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        d = Counter()
        l, res = 0, 0
        for r, c in enumerate(s):
            d[c] += 1
            while len(d) > k:
                d[s[l]] -= 1
                if d[s[l]] == 0:
                    d.pop(s[l])
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 487. Max Consecutive Ones II

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count, res, l = 0, 0, 0
        for r, n in enumerate(nums):
            if n == 0:
                count += 1
            while count > 1:
                if nums[l] == 0:
                    count -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 1100. Find K-Length Substrings With No Repeated Characters

```python
class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        if k > 26: return 0
        freq, res = [0] * 26, 0
        l, n = 0, len(s)
        for r in range(n):
            freq[ord(s[r]) - ord('a')] += 1
            while freq[ord(s[r]) - ord('a')] > 1:
                freq[ord(s[l]) - ord('a')] -= 1
                l += 1
            if r - l + 1 == k:
                res += 1
                freq[ord(s[l]) - ord('a')] -= 1
                l += 1
        return res
```

输入：
[[16,32],[27,3],[23,-14],[-32,-16],[-3,26],[-14,33]]
"aaabfc"
输出：
3
预期：
2