# Top 100 likes

## Note: 114, 394

## Hash table (3)

* [1. Two Sum](#1-Two-Sum) 1500
* [149. Group Anagrams](#49-Group-Anagrams) 1600
* [128. Longest Consecutive Sequence](#128-Longest-Consecutive-Sequence) 1700

## Two pointers (4)

* [11. Container With Most Water](#11-Container-With-Most-Water) 1600
* [15. 3Sum](#15-3Sum) 1700
* [283. Move Zeroes](#283-Move-Zeroes) 1500
* [42. Trapping Rain Water](#42-Trapping-Rain-Water) 1800

- Question list

## Hash table (3)

* [1. Two Sum](#1-Two-Sum) 1500
* [149. Group Anagrams](#49-Group-Anagrams) 1600
* [128. Longest Consecutive Sequence](#128-Longest-Consecutive-Sequence) 1700

## Sliding window (2)

* [3. Longest Substring Without Repeating Characters](#3-Longest-Substring-Without-Repeating-Characters) 1600
* [438. Find All Anagrams in a String](#438-Find-All-Anagrams-in-a-String) 1700


### 1. Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, v in enumerate(nums):
            res = target - v
            if res in d:
                return [d[res], i]
            d[v] = i
```

### 49. Group Anagrams

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for word in strs:
            d[''.join(sorted(list(word)))].append(word)
        return list(d.values())
```

### 128. Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        res = 0
        for n in nums:
            if n - 1 not in nums:
                j = n
                while j in nums:
                    j += 1
                res = max(res, j - n)
        return res
```

## Two pointers (4)

* [11. Container With Most Water](#11-Container-With-Most-Water) 1600
* [15. 3Sum](#15-3Sum) 1700
* [283. Move Zeroes](#283-Move-Zeroes) 1500
* [42. Trapping Rain Water](#42-Trapping-Rain-Water) 1800

### 11. Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res
```

### 15. 3Sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        s = set()
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, n - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if three == 0:
                    s.add((nums[i], nums[l], nums[r]))
                    l += 1
                    r -= 1
                elif three > 0:
                    r -= 1
                else:
                    l += 1
        return list(s)
```

### 283. Move Zeroes

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = 0
        for r, n in enumerate(nums):
            if n:
                nums[l] = n 
                l += 1
        for i in range(l, len(nums)):
            nums[i] = 0
```

### 42. Trapping Rain Water

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = 0
        for r, n in enumerate(nums):
            if n:
                nums[l] = n 
                l += 1
        for i in range(l, len(nums)):
            nums[i] = 0
```

## Sliding window (2)

* [3. Longest Substring Without Repeating Characters](#3-Longest-Substring-Without-Repeating-Characters) 1600
* [438. Find All Anagrams in a String](#438-Find-All-Anagrams-in-a-String) 1700

### 3. Longest Substring Without Repeating Characters

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l, res = 0, 0
        d = defaultdict(int)
        for r, c in enumerate(s):
            d[c] += 1
            while d[c] > 1:
                d[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res 
```

### 438. Find All Anagrams in a String

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        countP = Counter(p)
        res, l = [], 0
        d = Counter()
        P = len(p)
        for r, c in enumerate(s):
            d[c] += 1
            if r - l + 1 == P:
                if d == countP:
                    res.append(l)
                d[s[l]] -= 1
                l += 1
        return res 
```

## Backtracking

* [46. Permutations](#46-Permutations) 1600
* [78. Subsets](#78-Subsets) 1700
* [17. Letter Combinations of a Phone Number](#17-Letter-Combinations-of-a-Phone-Number) 1700
* [39. Combination Sum](#39-Combination-Sum) 1700
* [22. Generate Parentheses](#22-Generate-Parentheses) 1700
* [131. Palindrome Partitioning](#131-Palindrome-Partitioning) 1700
* [79. Word Search](#79-Word-Search) 1800
* [51. N-Queens](#51-N-Queens) 1800

### 46. Permutations

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, path):
            if not nums:
                res.append(path)
                return 
            for i in range(len(nums)):
                backtrack(nums[ :i] + nums[i + 1:], path + [nums[i]])
        res = []
        backtrack(nums, [])
        return res
```

### 78. Subsets

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(i, ans):
            if i == n:
                res.append(ans)
                return 
            backtrack(i + 1, ans + [nums[i]])
            backtrack(i + 1, ans)
        res, n = [], len(nums)
        backtrack(0, [])
        return res 
```

### 17. Letter Combinations of a Phone Number

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        d = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        def backtrack(i, cur):
            if len(cur) == n:
                res.append(cur)
                return
            for c in d[digits[i]]:
                backtrack(i + 1, cur + c)
        res, n = [], len(digits)
        backtrack(0, '')
        return res if digits else []
```

### 39. Combination Sum

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(idx, ans, total):
            if total == target:
                res.append(ans)
                return 
            if total > target:
                return 
            for i in range(idx, n):
                backtrack(i, ans + [candidates[i]], total + candidates[i])
        res, n = [], len(candidates)
        backtrack(0, [], 0)
        return res
```

### 22. Generate Parentheses

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def backtrack(open, close, ans):
            if open == close == n:
                res.append(ans)
                return 
            if open < n:
                backtrack(open + 1, close, ans + '(')
            if close < open:
                backtrack(open, close + 1, ans + ')')
        backtrack(0, 0, '')
        return res
```

### 131. Palindrome Partitioning

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def valid(s):
            return s == s[::-1]
        def backtrack(i, ans):
            if ans and not valid(ans[-1]):
                return 
            if i == n:
                res.append(ans)
                return 
            for j in range(i, n):
                backtrack(j + 1, ans + [s[i: j + 1]])

        n = len(s)
        res = []
        backtrack(0, [])
        return res 
```

### 79. Word Search

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        R, C, n, visited = len(board), len(board[0]), len(word), set()
        s = ''.join([''.join(item) for item in board])
        if any(s.count(c) < word.count(c) for c in word):
            return False
        def dfs(i, r, c):
            if i == len(word):
                return True
            if 0 <= r < R and 0 <= c < C and (r, c) not in visited and board[r][c] == word[i]:
                visited.add((r, c))
                res = dfs(i + 1, r + 1, c) or dfs(i + 1, r, c + 1) or dfs(i + 1, r - 1, c) or dfs(i + 1, r, c - 1)
                visited.remove((r, c))
                return res 
            return False

        for r in range(R):
            for c in range(C):
                if board[r][c] == word[0] and dfs(0, r, c):
                    return True
        return False
```

### 51. N-Queens

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        col, posDiag, negDiag = set(), set(), set()
        res, board = [], [['.'] * n for i in range(n)]
        def backtrack(r):
            if r == n:
                res.append([''.join(row) for row in board])
                return
            for c in range(n):
                if c not in col and (r + c) not in posDiag and (r - c) not in negDiag:
                    col.add(c)
                    posDiag.add(r + c)
                    negDiag.add(r - c)
                    board[r][c] = 'Q'
                    backtrack(r + 1)
                    col.remove(c)
                    posDiag.remove(r + c)
                    negDiag.remove(r - c)
                    board[r][c] = '.'
        backtrack(0)
        return res
```

## Graph

* [200. Number of Islands](#200-Number-of-Islands) 1500
* [994. Rotting Oranges](#994-Rotting-Oranges) 1432
* [207. Course Schedule](#207-Course-Schedule) 1700
* [208. Implement Trie (Prefix Tree)](#208-Implement-Trie) 1700

### 200. Number of Islands

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def valid(s):
            return s == s[::-1]
        def backtrack(i, ans):
            if ans and not valid(ans[-1]):
                return 
            if i == n:
                res.append(ans)
                return 
            for j in range(i, n):
                backtrack(j + 1, ans + [s[i: j + 1]])

        n = len(s)
        res = []
        backtrack(0, [])
        return res 
```

### 994. Rotting Oranges

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def valid(s):
            return s == s[::-1]
        def backtrack(i, ans):
            if ans and not valid(ans[-1]):
                return 
            if i == n:
                res.append(ans)
                return 
            for j in range(i, n):
                backtrack(j + 1, ans + [s[i: j + 1]])

        n = len(s)
        res = []
        backtrack(0, [])
        return res 
```

### 207. Course Schedule

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegree = [0] * numCourses
        g = defaultdict(list)
        for u, v in prerequisites:
            indegree[u] += 1
            g[v].append(u)

        q = deque([i for i, v in enumerate(indegree) if v == 0])
        res = 0
        while q:
            node = q.popleft()
            res += 1
            for nei in g[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
        return res == numCourses
```

### 208. Implement Trie

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def valid(s):
            return s == s[::-1]
        def backtrack(i, ans):
            if ans and not valid(ans[-1]):
                return 
            if i == n:
                res.append(ans)
                return 
            for j in range(i, n):
                backtrack(j + 1, ans + [s[i: j + 1]])

        n = len(s)
        res = []
        backtrack(0, [])
        return res 
```

## Heap (3)

* [215. Kth Largest Element in an Array](#215-Kth-Largest-Element-in-an-Array) 1500
* [347. Top K Frequent Elements](#347-Top-K-Frequent-Elements) 1600
* [295. Find Median from Data Stream](#295-Find-Median-from-Data-Stream) 1700

### 215. Kth Largest Element in an Array

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        pq = []
        for n in nums:
            heappush(pq, -n)
        
        for i in range(k):
            res = -heappop(pq)
        return res
```

### 347. Top K Frequent Elements

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        pq = []
        for key, v in Counter(nums).items():
            heappush(pq, (-v, key))
        res = []
        for i in range(k):
            v, key = heappop(pq)
            res.append(key)
        return res
```

### 295. Find Median from Data Stream

```python
class MedianFinder:

    def __init__(self):
        self.a = []
        self.b = []

    def addNum(self, num: int) -> None:
        if len(self.a) != len(self.b):
            heappush(self.b, -heappushpop(self.a, num))
        else:
            heappush(self.a, -heappushpop(self.b, -num))

    def findMedian(self) -> float:
        return self.a[0] if len(self.a) != len(self.b) else (self.a[0] - self.b[0]) / 2.0
```

## stack (3)

* [20. Valid Parentheses](#20-Valid-Parentheses) 1300
* [155. Min Stack](#155-Min-Stack) 1500
* [394. Decode String](#394-Decode-String) 1700

### 20. Valid Parentheses

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if stack and ((stack[-1] == '(' and c == ')') or (stack[-1] == '[' and c == ']') or (stack[-1] == '{' and c == '}')):
                stack.pop()
            else:
                stack.append(c)
        return not stack
```

### 155. Min Stack

```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.mn = [inf]

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.mn.append(min(self.mn[-1], val))

    def pop(self) -> None:
        self.stack.pop()
        self.mn.pop()
    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.mn[-1]
```

### 394. Decode String

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for c in s:
            if c != ']':
                stack.append(c)
            else:
                s = ''
                while stack[-1].isalpha():
                    c = stack.pop()
                    s = c + s
                stack.pop()
                d = ''
                while stack and stack[-1].isdigit():
                    c = stack.pop()
                    d = c + d 
                stack.append(s * int(d))
        return ''.join(stack)
```

## Monotonic stack (2)

* [739. Daily Temperatures](#739-Daily-Temperatures) 1700
* [84. Largest Rectangle in Histogram](#84-Largest-Rectangle-in-Histogram) 1900

### 739. Daily Temperatures

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res, stack = [0] * n, []
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                j = stack.pop()
                res[j] = i - j
            stack.append(i)
        return res
```

### 84. Largest Rectangle in Histogram

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        res, stack, heights = 0, [-1], heights + [0]
        for i, h in enumerate(heights):
            while len(stack) > 1 and h < heights[stack[-1]]:
                j = stack.pop()
                res = max(res, (i - 1 - stack[-1]) * heights[j])
            stack.append(i)
        return res
```

## Linked list

* [2. Add Two Numbers](#2-Add-Two-Numbers)

### 2. Add Two Numbers

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        carry = 0
        while l1 or l2:
            val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            p.next = ListNode(val % 10)
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            p = p.next
            carry = val // 10
        if carry: p.next = ListNode(carry)
        return dummy.next
```



### 4. Median of Two Sorted Arrays

- binary search
- https://www.youtube.com/watch?v=LPFhl65R7ww&t=1224s
- https://www.youtube.com/watch?v=q6IEA26hvXc

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B, total = nums1, nums2, len(nums1) + len(nums2)
        half = total // 2
        if len(A) > len(B): A, B = B, A
        lengthA, lengthB = len(A), len(B)
        l, r = 0, lengthA - 1
        while True:
            i = (l + r) // 2
            j = half - i - 2
            leftA = A[i] if i >= 0 else -inf
            rightA = A[i + 1] if i + 1 < lengthA else inf
            leftB = B[j] if j >= 0 else -inf
            rightB = B[j + 1] if j + 1 < lengthB else inf
            if leftA <= rightB and leftB <= rightA:
                if total % 2:
                    return min(rightA, rightB)
                return (max(leftA, leftB) + min(rightA, rightB)) / 2
            elif leftA > rightB:
                r = i - 1
            else:
                l = i + 1
```

### 5. Longest Palindromic Substring

- two pointers spread

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def subLongestPalindrome(l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1: r]

        res = ''
        for i in range(len(s)):
            res = max(res, subLongestPalindrome(i, i), subLongestPalindrome(i, i + 1), key = len)
        return res
```

- dp

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for c in range(n)]
        res = s[0]
        for c in range(1, n):
            for r in range(c):
                if s[r] == s[c] and (c - r <= 2 or dp[r + 1][c - 1]):
                    dp[r][c] = True
                    if c - r + 1 > len(res):
                        res = s[r: c + 1]
        return res
```
### 6. Zigzag Conversion

- bucket sort

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows ==  1 or numRows >= len(s):
            return s
        bucket, flip, row = [[] for i in range(numRows)], -1, 0
        for c in s:
            bucket[row].append(c)
            if row == numRows - 1 or row == 0:
                flip *= -1
            row += flip
        for i, arr in enumerate(bucket):
            bucket[i] = ''.join(arr)
        return ''.join(bucket)
```

### 7. Reverse Integer

- math

```python
class Solution:
    def reverse(self, x: int) -> int:
        def reversePositiveNum(x):
            res = 0
            while x:
                res = res * 10 + x % 10
                x //= 10
            return res
        res = reversePositiveNum(x) if x > 0 else -reversePositiveNum(-x)
        return res if -2 ** 31 <= res <= 2 ** 31 + 1 else 0
```

### 8. String to Integer (atoi)

- string

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        res, sign, s = '', ['+', '-'], s.strip()
        for i, c in enumerate(s):
            if i == 0 and c in sign or c.isnumeric():
                res += c
            else:
                break
        if not res or res in sign:
            return 0
        return min(max(int(res), -2 ** 31), 2 ** 31 - 1)
```

### 9. Palindrome Number

- math

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        def reversePositiveNum(x):
            res = 0
            while x:
                res = res * 10 + x % 10
                x //= 10
            return res
        return reversePositiveNum(x) == x
```

### 10. Regular Expression Matching

- https://www.youtube.com/watch?v=HAA8mgxlov8
- dp

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        lengthS, lengthP = len(s), len(p)
        @lru_cache(None)
        def dfs(i, j):
            if i >= lengthS and j >= lengthP:
                return True
            if j >= lengthP:
                return False
            match = i < lengthS and (s[i] == p[j] or p[j] == '.')
            if j + 1 < lengthP and p[j + 1] == '*': # check *, 0 or more
                return dfs(i, j + 2) or (match and dfs(i + 1, j))
            return dfs(i + 1, j + 1) if match else False # check match or not if no *
        return dfs(0, 0)
```

## sweep line

### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        events = []
        for s, e in intervals:
            events.append((s, -1))
            events.append((e, 1))
        events.sort()

        res = []
        start = inf
        count = 0
        for t, sign in events:
            if sign == -1:
                count += 1
                start = min(start, t)
            else:
                count -= 1
                if count == 0:
                    res.append([start, t])
                    start = inf
        return res
```


## monotonic queue

### 239. Sliding Window Maximum

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res, q = [], deque()
        for r, n in enumerate(nums):
            while q and n > nums[q[-1]]:
                q.pop()
            q.append(r)
            if r - q[0] + 1 > k:
                q.popleft()
            if r >= k - 1:
                res.append(nums[q[0]])
        return res
```

### 189. Rotate Array

- need to prepare the video

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n 
        nums.reverse()
        l, r = 0, k - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
        l, r = k, n - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
```

### 238. Product of Array Except Self

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res, tmp = [1] * n, 1
        for i in range(1, n):
            res[i] = res[i - 1] * nums[i - 1]
        for i in range(n - 2, -1, -1):
            tmp *= nums[i + 1]
            res[i] *= tmp
        return res
```

### 234. Palindrome Linked List

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        res = []
        while head:
            res.append(head.val)
            head = head.next 
        return res == res[::-1]

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        total = 0
        p = head 
        while p:
            total += 1
            p = p.next 

        p = dummy = ListNode()
        p.next = head 
        slow, fast = dummy, dummy 
        while fast and fast.next:
            slow = slow.next 
            fast = fast.next.next 
        if total % 2 == 0:
            l2 = slow.next 
        else:
            l2 = slow 
        def reverseLinkedList(prev, cur):
            while cur:
                nxt = cur.next 
                cur.next = prev 
                prev, cur = cur, nxt 
            return prev 

        p2 = reverseLinkedList(None, l2)
        while head and p2:
            if head.val != p2.val:
                return False
            head = head.next 
            p2 = p2.next
        return True
```

### 148. Sort List

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        res = []
        while head:
            res.append(head.val)
            head = head.next 
        res.sort()
        dummy = p = ListNode()
        for n in res:
            p.next = ListNode(n)
            p = p.next 
        return dummy.next 
```

### 160. Intersection of Two Linked Lists

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        h1, h2 = headA, headB
        while h1 is not h2:
            h1 = h1.next if h1 else headB
            h2 = h2.next if h2 else headA
        return h1
```

## Moore Voting

### 169. Majority Element

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        for n in nums:
            if count == 0:
                major = n 
                count = 1
            elif n == major:
                count += 1
            else:
                count -= 1
        return major
```

## greedy

### 55. Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        furthest = 0
        for i in range(n):
            if i <= furthest:
                furthest = max(i + nums[i], furthest)
        return furthest >= n - 1
```

### 102. Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        res = []
        while q:
            a = []
            for i in range(len(q)):
                node = q.popleft()
                a.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(a)
        return res
```

## clone 

### 138. Copy List with Random Pointer

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        d = {}
        p = head 
        while p:
            d[p] = Node(p.val)
            p = p.next 

        p = head 
        while p:
            d[p].next = d.get(p.next)
            d[p].random = d.get(p.random)
            p = p.next 
        return d[head] if head else None
```

### 146. LRU Cache

```python
class ListNode:
    def __init__(self, key = 0, value = 0):
        self.key = key
        self.value = value

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.head = self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = ListNode(key, value)
            self.cache[key] = node
            self.addToHead(node)
            self.capacity -= 1
            if self.capacity < 0:
                removed = self.removeTail(self.tail.prev)
                self.cache.pop(removed.key)
        else:
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeTail(self, node):
        self.removeNode(node)
        return node
```

### 108. Convert Sorted Array to Binary Search Tree

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(l, r):
            if l > r:
                return None
            m = l + (r - l) // 2
            root = TreeNode(nums[m])
            root.left = dfs(l, m - 1)
            root.right = dfs(m + 1, r)
            return root
        return dfs(0, len(nums) - 1)
```

#### 109. Convert Sorted List to Binary Search Tree

```python
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
            
        def dfs(l, r):
            if l > r:
                return None
            m = l + (r - l) // 2
            root = TreeNode(nums[m])
            root.left = dfs(l, m - 1)
            root.right = dfs(m + 1, r)
            return root
        return dfs(0, len(nums) - 1)
```

### 98. Validate Binary Search Tree

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode], l = -inf, r = inf) -> bool:
        if not root:
            return True
        v = root.val 
        if v <= l or v >= r:
            return False
        return self.isValidBST(root.left, l, v) and self.isValidBST(root.right, v, r)
```

### 198. House Robber

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [0] * (len(nums) + 1)
        dp[1] = nums[0]
        for i in range(2, len(dp)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[-1]
```

### 322. Coin Change

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        @cache
        def dfs(t, i):
            if t > amount:
                return inf 
            if i == len(coins):
                return 0 if t == amount else inf 
            return min(dfs(t, i + 1), dfs(t + coins[i], i) + 1)
        res = dfs(0, 0)
        return res if res != inf else -1
```

### 279. Perfect Squares

```python
square = [i * i for i in range(100, 0, -1)]
class Solution:
    def numSquares(self, n: int) -> int:
        @cache
        def f(t, i):
            if t > n:
                return inf 
            if i == len(square):
                return 0 if t == n else inf
            return min(f(t, i + 1), f(t + square[i], i) + 1) 
        res = f(0, 0)
        f.cache_clear()
        return res
```

