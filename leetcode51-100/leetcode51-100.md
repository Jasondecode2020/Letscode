
### 51. N-Queens

- backtracking

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
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue
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

### 52. N-Queens II

- backtracking

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        col, posDiag, negDiag = set(), set(), set()
        res, board = 0, [['.'] * n for i in range(n)]
        def backtrack(r):
            if r == n:
                nonlocal res
                res += 1
                return

            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue
                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                backtrack(r + 1)
                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
        backtrack(0)
        return res
```

### 53. Maximum Subarray (dp, 978)

- dp

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp, res = [nums[0]] * n, nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
            res = max(res, dp[i])
        return res
```

### 54. Spiral Matrix

- matrix

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        while matrix:
            res.extend(matrix.pop(0))
            matrix = list(zip(*matrix))[::-1]
        return res
```

### 55. Jump Game

- dp 

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        res, n = 0, len(nums)
        for i in range(n):
            res = max(res, i + nums[i])
            if res >= n - 1:
                return True
            elif res <= i:
                return False
```

### 56. Merge Intervals (57)

- greedy

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= res[-1][1]:
                res[-1][1] = max(end, res[-1][1])
            else:
                res.append([start, end])
        return res
```

### 57. Insert Interval (56)

- greedy

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        intervals.sort()
        res = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= res[-1][1]:
                res[-1][1] = max(end, res[-1][1])
            else:
                res.append([start, end])
        return res
```

### 58. Length of Last Word

- string

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])
```

### 59. Spiral Matrix II

- matrix

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        A, lo = [], n*n+1
        while lo > 1:
            lo, hi = lo - len(A), lo
            A = [range(lo, hi)] + list(zip(*A[::-1]))
        return A
```

### 60. Permutation Sequence 

- math

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        res, nums = '', list(range(1, n + 1))
        for i in range(1, n + 1):
            index = 0
            cnt = factorial(n - i)
            while k > cnt:
                index += 1
                k -= cnt
            res += str(nums[index])
            nums.pop(index)
        return res
```

### 61. Rotate List
 
- linked list

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head: return head
        #connect tail to head
        cur, length = head, 1
        while cur.next:
            cur = cur.next
            length+=1 
        cur.next = head

        # move to new head
        k = length - (k % length)
        while k > 0:
            cur=cur.next
            k-=1

        #disconnect and return new head
        newhead = cur.next
        cur.next = None
        return newhead
```

### 62. Unique Paths

- dp, math

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # # method 1: dp
        '''
        ROWS, COLS = m, n
        dp = [[1] * COLS for i in range(ROWS)]
        for i in range(1, ROWS):
            for j in range(1, COLS):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]
        '''
        
        # method 2: dfs
        '''
        @lru_cache(None)
        def dfs(row, col):
            if row == m or col == n:
                return 0
            if row == m - 1 and col == n - 1:
                return 1
            return dfs(row + 1, col) + dfs(row, col + 1)
        return dfs(0, 0)
        '''

        ### method 3: math
        return factorial(m + n - 2) // (factorial(n - 1) * factorial(m - 1))
```

### 63. Unique Paths II
 
- dp

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # method 1: dfs + memo
        '''
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        @lru_cache(None)
        def dfs(row, col):
            if row == m or col == n or obstacleGrid[row][col]:
                return 0
            if row == m - 1 and col == n - 1:
                if obstacleGrid[row][col]:
                    return 0
                return 1
            return dfs(row + 1, col) + dfs(row, col + 1)
        return dfs(0, 0)
        '''
        
        # method 2: dp
        ROWS, COLS = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[1] * COLS for i in range(ROWS)]
        if obstacleGrid[0][0]: return 0
        for j in range(1, COLS):
            dp[0][j] = 0 if obstacleGrid[0][j] else dp[0][j - 1]
        for i in range(1, ROWS):
            dp[i][0] = 0 if obstacleGrid[i][0] else dp[i - 1][0]
        for i in range(1, ROWS):
            for j in range(1, COLS):
                if obstacleGrid[i][j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]
```

### 64. Minimum Path Sum

- dp 

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        R, C, first = len(grid), len(grid[0]), grid[0][0]
        # state
        dp = [[first] * C for i in range(R)]
        # init rows
        for j in range(1, C):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        # init cols
        for i in range(1, R):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        # top-down
        for i in range(1, R):
            for j in range(1, C):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]
```

### 65. Valid Number

- math 

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        # s = s.strip()
        met_dot = met_e = met_digit = False
        for i, char in enumerate(s):
            if char in ['+', '-']:
                if i > 0 and s[i-1].lower() != 'e':
                    return False
            elif char == '.':
                if met_dot or met_e: return False
                met_dot = True
            elif char.lower() == 'e':
                if met_e or not met_digit:
                    return False
                met_e, met_digit = True, False
            elif char.isdigit():
                met_digit = True
            else:
                return False
        return met_digit
```

### 66. Plus One

- math 

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carry, digits[-1] = 0, digits[-1] + 1
        for i in range(len(digits) - 1, -1, -1):
            res = carry + digits[i]
            digits[i] = res % 10
            carry = res // 10
        return [carry] + digits if carry else digits
```

### 67. Add Binary

- math

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        carry, i, j, res = 0, len(a) - 1, len(b) - 1, ''
        while i >= 0 or j >= 0:
            ans = carry
            if i >= 0:
                ans += ord(a[i]) - ord('0')
            if j >= 0:
                ans += ord(b[j]) - ord('0')
            i, j = i - 1, j - 1
            carry = 1 if ans > 1 else 0
            res = str(ans % 2) + res
        if carry:
            res = str(carry) + res
        return res
```

### 68. Text Justification

- math

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res, i, width, cur_line = [], 0, 0, []
        while i < len(words):
            cur_word = words[i]
            if width + len(cur_word) <= maxWidth:
                cur_line.append(cur_word)
                width += len(cur_word) + 1
                i += 1
            else:
                spaces = maxWidth - width + len(cur_line) # check how many spaces
                added, j = 0, 0
                while added < spaces: # put the spaces between the words evenly
                    if j >= len(cur_line) - 1:
                        j = 0
                    cur_line[j] += ' '
                    added, j = added + 1, j + 1
                res.append("".join(cur_line))
                cur_line, width = [], 0

        # handle last line
        for word in range(len(cur_line) - 1):
            cur_line[word] += ' '
        cur_line[-1] += ' ' * (maxWidth - width + 1)
        res.append(''.join(cur_line))
        return res
```

### 69. Sqrt(x)

- bianry search

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        l, r = 0, x
        while l <= r:
            m = l + (r - l) // 2
            if m * m < x:
                l = m + 1
            elif m * m > x:
                r = m - 1
            else:
                return m
        return r
```

### 70. Climbing Stairs

- dp

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        first, second = 1, 2
        for i in range(3, n + 1):
            second, first = second + first, second
        return second if n >= 2 else first
```

### 71. Simplify Path

- dp

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        for item in path.split('/'):
            if item in ('', '.'):
                continue
            if item == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(item)
        return '/' + '/'.join(stack)
```

### 72. Edit Distance

- dp

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len1, len2 = len(word1), len(word2)
        dp = [[0] * (len2 + 1) for i in range(len1 + 1)]
        for i in range(1, len1 + 1): dp[i][0] = i
        for j in range(1, len2 + 1): dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, 
                    dp[i - 1][j - 1] if word1[i - 1] == word2[j - 1] else dp[i - 1][j - 1] + 1)
        return dp[-1][-1]
```

### 73. Set Matrix Zeroes

- set

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        R, C, row, col = len(matrix), len(matrix[0]), set(), set()
        for r in range(R):
            for c in range(C):
                if matrix[r][c] == 0:
                    row.add(r)
                    col.add(c)
        for r in range(R):
            for c in range(C):
                if r in row or c in col:
                    matrix[r][c] = 0
```

### 74. Search a 2D Matrix

- binary search

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        def search(nums, n):
            l, r = 0, len(nums) - 1
            while l <= r:
                m = l + (r - l) // 2
                if nums[m] > n:
                    r = m - 1
                elif nums[m] < n:
                    l = m + 1
                else:
                    return True
            return False
        # check column if not exist, leave it row, row will return false
        # if find, leave it to row, if exist, it will become true
        l, r = 0, len(matrix) - 1
        while l <= r:
            m = l + (r - l) // 2
            if matrix[m][0] > target:
                r = m - 1
            else:
                l = m + 1
            
        if search(matrix[l - 1], target):
            return True
        return False
```

### 75. Sort Colors

- three pointers

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] > nums[j]:
                    nums[i], nums[j] = nums[j], nums[i]
```
Follow up: Could you come up with a one-pass algorithm using only constant extra space?

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i, l, r = 0, 0, len(nums) - 1
        while i <= r:
            if nums[i] == 0:
                nums[i], nums[l] = nums[l], nums[i]
                l += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[r] = nums[r], nums[i]
                r -= 1
            else:
                i += 1
```

### 76. Minimum Window Substring

- sliding windows

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        res, l, tCounter, window = "", 0, Counter(t), Counter()
        for r, c in enumerate(s):
            window[c] += 1
            while window >= tCounter:
                if res == "" or r - l + 1 < len(res):
                    res = s[l: r + 1]
                window[s[l]] -= 1
                l += 1
        return res
```

### 77. Combinations

- backtracking

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        def backtrack(start, ans):
            if len(ans) == k:
                res.append(ans[::])
                return
            for i in range(start, n + 1):
                ans.append(i)
                backtrack(i + 1, ans)
                ans.pop()
        backtrack(1, [])
        return res
```

### 78. Subsets

- backtracking

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res, subset, n = [], [], len(nums)
        def dfs(i):
            if i == n:
                res.append(subset[::])
                return
            # add num
            subset.append(nums[i])
            dfs(i + 1)
            # add no num
            subset.pop()
            dfs(i + 1)
        dfs(0)
        return res
```

### 79. Word Search

- backtracking

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS = len(board)
        COLS = len(board[0])
        visit = set()
        # B_str = ''.join([''.join(i) for i in board])
        # prune the False cases before dfs
        # if any(B_str.count(i) < word.count(i) for i in set(word)):
        #     return False
        def dfs(r, c, i):
            if i == len(word):
                return True
            if r < 0 or c < 0 or r >= ROWS or c >= COLS or board[r][c] != word[i] or \
                    (r, c) in visit:
                return False
            visit.add((r, c))
            res = dfs(r - 1, c, i + 1) or dfs(r + 1, c, i + 1) or \
                    dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1)
            visit.remove((r, c))
            return res
        
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True
        return False
```

method 2:

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS = len(board)
        COLS = len(board[0])
        visit = set()
        # B_str = ''.join([''.join(i) for i in board])
        # # prune the False cases before dfs
        # if any(B_str.count(i) < word.count(i) for i in word):
        #     return False
        def dfs(r, c, i):
            if i == len(word):
                return True
            if 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == word[i] and (r, c) not in visit:
                visit.add((r, c))
                res = dfs(r - 1, c, i + 1) or dfs(r + 1, c, i + 1) or \
                        dfs(r, c + 1, i + 1) or dfs(r, c - 1, i + 1)
                visit.remove((r, c))
                return res
            return False
        
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True
        return False
```

### 80. Remove Duplicates from Sorted Array II

- array

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        idx = 2
        for i in range(2, len(nums)):
            if nums[i] != nums[idx - 2]:
                nums[idx] = nums[i]
                idx += 1
        return idx
```

### 81. Search in Rotated Sorted Array II

- binary + linear: can be improved

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        def binary(arr, target):
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return True
                elif arr[mid] > target:
                    hi -= 1
                else:
                    lo += 1
            return False
        if nums[0] == target: return True
        for i in range(1, len(nums)):
            if nums[i] == target:
                return True
            if nums[i] < nums[i-1]:
                return binary(nums[i:], target)
        return False
```

### 82. Remove Duplicates from Sorted List II - medium

- linked list

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        p = dummy = ListNode(200)
        dummy.next = head
        hash_set = set()
        while p and p.next:
            if p.val != p.next.val:
                p = p.next
            else:
                hash_set.add(p.val)
                p.next = p.next.next
        print(hash_set)
        q = dummy
        while q.next:
            if q.next.val in hash_set:
                q.next = q.next.next
            else:
                q = q.next
        return dummy.next
```

### 83. Remove Duplicates from Sorted List - easy

- linked list

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p = head
        while p and p.next:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next
        return head
```

### 84. Largest Rectangle in Histogram - hard

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        res, stack = 0, []
        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                res = max(res, height * (i - index))
                start = index
            stack.append((start, h))
        for i, h in stack:
            res = max(res, h * (len(heights) - i))
        return res
```

### 85. Maximal Rectangle - hard

- stack

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def histHelper(heights):
            res, stack = 0, []
            for i, h in enumerate(heights):
                start = i
                while stack and stack[-1][1] > h:
                    index, height = stack.pop()
                    res = max(res, height * (i - index))
                    start = index
                stack.append((start, h))
            # handle remaining stack:
            for i, h in stack:
                res = max(res, h * (len(heights) - i))
            return res
        # calculate each row
        res = 0
        heights = [0] * len(matrix[0])
        for row in matrix:
            for i in range(len(heights)):
                heights[i] = heights[i] + 1 if row[i] == '1' else 0
            res = max(res, histHelper(heights))
        return res
```

### 86. Partition List

- linked list

```python
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        # 2 pass
        r = dummy = ListNode(0)
        p = q = head
        while p:
            if p.val < x:
                r.next = ListNode(p.val)
                r = r.next
            p = p.next
        while q:
            if q.val >= x:
                r.next = ListNode(q.val)
                r = r.next
            q = q.next
        return dummy.next
```

### 87. Scramble String - hard

- dp

```python
class Solution:
    @cache
    def isScramble(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        L = len(s1)
        for k in range(1, L):
            if self.isScramble(s1[0:k], s2[0:k]) and self.isScramble(s1[k:], s2[k:]):
                return True
            if self.isScramble(s1[0:k], s2[L-k:]) and self.isScramble(s1[k:], s2[0:L-k]):
                return True
        return False
```

### 88. Merge Sorted Array

- array

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i, j, idx = m - 1, n - 1, m + n - 1
        while j >= 0:
            if i >= 0 and nums1[i] >= nums2[j]:
                nums1[idx] = nums1[i]
                i -= 1
            else:
                nums1[idx] = nums2[j]
                j -= 1
            idx -= 1
```

### 89. Gray Code

- dfs

```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        def helper(n):
            if n == 0:
                return ['0']
            if n == 1:
                return ['0', '1']
            res = helper(n - 1)
            return ['0' + i for i in res] + ['1' + i for i in res[::-1]]
        return [int(i, 2) for i in helper(n)]
```

### 90. Subsets II - medium

- backtracking - same as 78. Subsets I

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res, subset, n = set(), [], len(nums)
        def dfs(i):
            if i == n:
                res.add(tuple(subset[::]))
                return
            # add num
            subset.append(nums[i])
            dfs(i + 1)
            # add no num
            subset.pop()
            dfs(i + 1)
        dfs(0)
        return res
```

### 91. Decode Ways

- dfs

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        @lru_cache(None)
        def dfs(i):
            if i == n:
                return 1
            if s[i] == '0':
                return 0
            res = dfs(i + 1)
            if i + 1 < n and 10 <= int(s[i: i + 2]) <= 26:
                res += dfs(i + 2)
            return res
        return dfs(0)
```

### 92. Reverse Linked List II

- linked list

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        prev = dummy = ListNode(0)
        dummy.next = head
        i = 1
        while i < left:
            prev = prev.next
            i += 1
        curr = prev.next
        while left < right:
            next = curr.next
            curr.next = next.next
            next.next = prev.next
            prev.next = next
            left += 1
        return dummy.next
```

### 93. Restore IP Addresses

- backtracking - same as 78. Subsets I

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res, n = [], len(s)
        if len(s) > 12:
            return res
        def backtrack(i, dots, curIP):
            if dots == 4 and i == n:
                res.append(curIP[: -1])
                return
            if dots > 4 or i == n:
                return
            for j in range(i, min(i + 3, n)):
                if int(s[i: j + 1]) < 256 and (i == j or s[i] != '0'):
                    backtrack(j + 1, dots + 1, curIP + s[i: j + 1] + '.')
        backtrack(0, 0, '')
        return res
```

### 94. Binary Tree Inorder Traversal

```python
# recursive
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node, arr):
            if not node:
                return node
            dfs(node.left, arr)
            arr.append(node.val)
            dfs(node.right, arr)
            return arr
        return dfs(root, [])
# iterative
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res, stack = [], []
        while True:
            while root:
                stack.append(root)
                root = root.left
            if not stack:
                return res
            node = stack.pop()
            res.append(node.val)
            root = node.right
```

### 95. Unique Binary Search Trees II

- dfs

```python
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        @cache
        def dfs(start, end):
            if start > end:
                return [None]
            ans = []
            for i in range(start, end + 1):
                left = dfs(start, i - 1)
                right = dfs(i + 1, end)
                for l in left:
                    for r in right:
                        root = TreeNode(i)
                        root.left = l
                        root.right = r
                        ans.append(root)
            return ans
        return dfs(1, n)
```

### 96. Unique Binary Search Trees

- dfs + memo

```python
class Solution:
    def numTrees(self, n: int) -> int:
        @lru_cache(None)
        def dfs(num):
            if num <= 1:
                return 1
            res = 0
            for i in range(num):
                res += dfs(i) * dfs(num - i - 1)
            return res
        return dfs(n)
```

### 97. Interleaving String

- dfs

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if (len(s1 + s2) != len(s3)): return False
        @cache
        def dfs(i, j):
            if (i == len(s1) and j == len(s2)):
                return True
            if i < len(s1) and s1[i] == s3[i + j] and dfs(i + 1, j):
                return True
            if j < len(s2) and s2[j] == s3[i + j] and dfs(i, j + 1):
                return True
            return False
        return dfs(0, 0)
```

### 98. Validate Binary Search Tree

- Tree

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, left, right):
            if not node:
                return True
            if node.val <= left or node.val >= right:
                return False
            return dfs(node.left, left, node.val) and dfs(node.right, node.val, right)
        return dfs(root, -inf, inf)
```

### 99. Recover Binary Search Tree

- Tree

```python
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def inorder(node, res):
            if node:
                inorder(node.left, res)
                res.append(node)
                inorder(node.right, res)
            return res
        nodes = inorder(root, [])
        nodes = [TreeNode(-inf)] + nodes + [TreeNode(inf)]
        res = sorted([node.val for node in nodes])
        for i, node in enumerate(nodes):
            if node.val != res[i]:
                node.val = res[i]
```

### 100. Same Tree

- tree

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```
