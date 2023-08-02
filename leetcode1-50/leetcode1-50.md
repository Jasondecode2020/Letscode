### 1. Two Sum

- hash table

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

### 2. Add Two Numbers

- linked list

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

### 3. Longest Substring Without Repeating Characters

- sliding window

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res, l, d = 0, 0, {}
        for r, v in enumerate(s):
            if v in d:
                l = max(l, d[v] + 1)
            res = max(res, r - l + 1)
            d[v] = r
        return res
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

- brute force

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

- string

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

- dp
- https://www.youtube.com/watch?v=HAA8mgxlov8

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

### 11. Container With Most Water

- two pointers

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r, res = 0, len(height) - 1, 0
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return res
```

### 12. Integer to Roman

- math

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        numbers = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        res = ''
        for i, n in enumerate(numbers):
            res += num // n * symbols[i]
            num %= n
        return res
```

### 13. Roman to Integer

- math 

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        d = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        res = d[s[0]]
        for i in range(1, len(s)):
            res += d[s[i]]
            if d[s[i]] > d[s[i - 1]]:
                res -= 2 * d[s[i - 1]]
        return res
```

### 14. Longest Common Prefix

- trie

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = strs[0]
        for item in strs:
            while not item.startswith(res):
                res = res[: -1]
        return res
```

#### Tag: Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True
        
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        trie = Trie()
        for word in strs:
            trie.insert(word)
            
        root = trie.root
        res = ''
        while root:
            if len(root.children) > 1 or root.endOfWord:
                break
            c = list(root.children)[0]
            root = root.children[c]
            res += c
        return res
```

### 15. 3Sum

- two pointers

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res, n = set(), len(nums)
        for i in range(n):
            l, r = i + 1, n - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if three > 0:
                    r -= 1
                elif three < 0:
                    l += 1
                else:
                    res.add((nums[i], nums[l], nums[r]))
                    l += 1
                    r -= 1
        return res
```

### 16. 3Sum Closest

- two pointers

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        res, n = float('inf'), len(nums)
        nums.sort()
        for i in range(n):
            l, r = i + 1, n - 1
            while l < r:
                three = nums[i] + nums[l] + nums[r]
                if abs(target - three) < abs(target - res):
                    res = three
                if three > target:
                    r -= 1
                else:
                    l += 1
        return res
```

### 17. Letter Combinations of a Phone Number

- string

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
        res = ['']
        for n in digits:
            letters = d[n]
            ans = []
            for item in res:
                for c in letters:
                    ans.append(item + c)
            res = ans
        return res if digits else []
```

### 18. 4Sum

- two pointers

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        s, n = set(), len(nums)
        nums.sort()
        for i in range(n):
            for j in range(i + 1, n):
                l, r = j + 1, n - 1
                while l < r:
                    four = nums[i] + nums[j] + nums[l] + nums[r]
                    if four == target:
                        s.add((nums[i], nums[j], nums[l], nums[r]))
                        l += 1
                        r -= 1
                    elif four < target:
                        l += 1
                    else:
                        r -= 1
        return s
```

### 19. Remove Nth Node From End of List

- linked list

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = slow = fast = ListNode()
        dummy.next = head
        for i in range(n + 1):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

### 20. Valid Parentheses

- stack

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack, valid = [], ['{}', '[]', '()']
        for c in s:
            if stack and any([c == valid[i][1] and stack[-1] == valid[i][0] for i in range(len(valid))]):
                stack.pop()
            else:
                stack.append(c)
        return not stack
```

### 21. Merge Two Sorted Lists

- linked list

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                p.next = ListNode(list1.val)
                list1 = list1.next
            else:
                p.next = ListNode(list2.val)
                list2 = list2.next
            p = p.next
        p.next = list1 if list1 else list2
        return dummy.next
```

### 22. Generate Parentheses

- backtracking

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        stack, res = [], []
        def backtrack(open, close):
            if open == close == n:
                res.append(''.join(stack))
                return
            if open < n:
                stack.append('(')
                backtrack(open + 1, close)
                stack.pop()
            if close < open:
                stack.append(')')
                backtrack(open, close + 1)
                stack.pop()
        backtrack(0, 0)
        return res
```

### 23. Merge k Sorted Lists

- linked list

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        dummy = p = ListNode()
        minHeap = []
        for head in lists:
            while head:
                heappush(minHeap, head.val)
                head = head.next
        while minHeap:
            val = heappop(minHeap)
            p.next = ListNode(val)
            p = p.next
        return dummy.next
```

### 24. Swap Nodes in Pairs

- linked list

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = dummy = ListNode(0)
        prev.next, curr = head, head
        total = 0
        while head:
            total += 1
            head = head.next

        def reverseOnePair():
            nxt = curr.next
            curr.next, nxt.next, prev.next = nxt.next, prev.next, nxt
        for i in range(total // 2):
            reverseOnePair()
            prev, curr = curr, curr.next
        return dummy.next
```

### 25. Reverse Nodes in k-Group

- linked list

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        prev = dummy = ListNode(0)
        prev.next, curr = head, head
        total = 0
        while head:
            total += 1
            head = head.next
            
        def reverseOnePair():
            nxt = curr.next
            curr.next, nxt.next, prev.next = nxt.next, prev.next, nxt
        for i in range(total // k):
            for j in range(k - 1):
                reverseOnePair()
            prev, curr = curr, curr.next
        return dummy.next
```

### 26. Remove Duplicates from Sorted Array

- array

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[k] = nums[i]
                k += 1
        return k
```

### 27. Remove Element

- two pointers

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k, r = 0, len(nums) - 1
        while k <= r:
            if nums[k] == val:
                nums[k] = nums[r]
                r -= 1
            else:
                k += 1
        return k
```

### 28. Find the Index of the First Occurrence in a String

- Robin Karp: rolling hash

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n, pattern = len(needle), hash(needle)
        for i in range(len(haystack)):
            if hash(haystack[i:i+n]) == pattern:
                return i
        return -1
```

### 29. Divide Two Integers

- bit manipulation

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if (dividend == -2 ** 31 and divisor == -1): return 2 ** 31 - 1 # max value overflow
        a, b, res = abs(dividend), abs(divisor), 0
        for x in range(32)[::-1]:
            if (a >> x) - b >= 0: # first: 1010 >> 1 = 101 > 11, second: 100 >> 0 = 100 > 11
                res += 1 << x # first: 0 + 1 << 1 = 2, second: 2 + 1 << 0 = 2 + 1 = 3
                a -= b << x # first: 1010 - 11 << 1 = 10 - 6 = 4, second: 100 - 11 << 0 = 100 - 11 = 1, stop
        return res if (dividend > 0) == (divisor > 0) else -res
```

### 30. Substring with Concatenation of All Words

- string

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        ans, n, m, words  = [], len(words), len(words[0]), Counter(words)
        for i in range(len(s) - n * m + 1):
            tmp, cnt = Counter(), 0
            for j in range(i, i + n * m, m):
                w = s[j: j + m]
                if w in words:
                    tmp[w] += 1
                    cnt += 1
                    if tmp[w] > words[w]: 
                        break
                    if cnt == n:
                        ans.append(i)
        return ans
```

### 31. Next Permutation

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 1 find pivot
        pivot = None
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] > nums[i - 1]:
      
                pivot = i - 1
                break
        if pivot == None:
            nums.reverse()
            return
        # 2 swap with pivot if larger than pivot
        print(nums)
        for i in range(len(nums) - 1, pivot, -1):
            if nums[i] > nums[pivot]:
                nums[i], nums[pivot] = nums[pivot], nums[i]
                break
        # 3 reverse nums after pivot
        nums[pivot +1: ] = nums[pivot +1: ][::-1]
```

### 32. Longest Valid Parentheses

- stack
https://www.youtube.com/watch?v=q56S5NIqjdE

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack, res = [-1], 0 # edge cases of s = "()"
        for i in range(len(s)):
            if s[i] == "(": # prepare for finding the max
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res
```

### 33. Search in Rotated Sorted Array

- binary search

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            if nums[m] >= nums[l]:
                if (nums[l] <= target < nums[m]):
                    r = m - 1
                else:
                    l = m + 1
            else:
                if (nums[m] < target <= nums[r]):
                    l = m + 1
                else:
                    r = m - 1
        return -1
```

### 34. Find First and Last Position of Element in Sorted Array

- binary search

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        def binary_right(nums, target):
            l, r = 0, n - 1
            while l <= r:
                m = l + (r - l) // 2
                if nums[m] > target:
                    r = m - 1
                else:
                    l = m + 1
            return r if r >= 0 and nums[r] == target else -1
        def binary_left(nums, target):
            l, r = 0, n - 1
            while l <= r:
                m = l + (r - l) // 2
                if nums[m] < target:
                    l = m + 1
                else:
                    r = m - 1
            return l if l < n and nums[l] == target else -1
        return [binary_left(nums, target), binary_right(nums, target)]
```

### 35. Search Insert Position

- binary search

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect_left(nums, target)
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                return m
        return l
```

### 36. Valid Sudoku

- set

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows, cols, squres = defaultdict(set), defaultdict(set), defaultdict(set)
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] in (rows[r] | cols[c] | squres[(r // 3, c // 3)]):
                    return False
                if board[r][c] != '.':
                    rows[r].add(board[r][c])
                    cols[c].add(board[r][c])
                    squres[(r // 3, c // 3)].add(board[r][c])
        return True
```

### 37. Sudoku Solver

- backtracking 

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows, cols, squares, visit = defaultdict(set), defaultdict(set), defaultdict(set), deque([])
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    squares[(i // 3, j // 3)].add(board[i][j])
                else:
                    visit.append((i,j))

        def dfs():
            if not visit:
                return True

            r, c = visit[0]
            square, numbers = (r // 3, c // 3), {'1','2','3','4','5','6','7','8','9'}
            for n in numbers: # try 9 ways
                if n not in (rows[r] | cols[c] | squares[square]):
                    board[r][c] = n
                    rows[r].add(n)
                    cols[c].add(n)
                    squares[square].add(n)
                    visit.popleft()
                    if dfs(): # find 1 way
                        return True
                    else: # backtrack
                        board[r][c] = "."
                        rows[r].discard(n)
                        cols[c].discard(n)
                        squares[square].discard(n)
                        visit.appendleft((r,c))
            return False # not find
        dfs()
```

### 38. Count and Say

- recursion

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        if n == 1:
            return '1'
        preString= self.countAndSay(n-1)
        count, res, length = 1, '', len(preString)
        for i in range(length):
            if i < length - 1 and preString[i] == preString[i + 1]:
                count += 1
            else:
                res += str(count) + preString[i]
                count = 1
        return res
```

### 39. Combination Sum

- backtracking

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res, n = [], len(candidates)
        def backtrack(curr, ans, idx):
            if curr > target:
                return
            if curr == target:
                res.append(ans)
            for i in range(idx, n):
                backtrack(curr + candidates[i], [candidates[i]] + ans, i)
        backtrack(0, [], 0)
        return res
```

### 40. Combination Sum II

- backtracking

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        def backtrack(nums, ans, curr):
            if curr > target:
                return
            if curr == target:
                res.append(ans)
            for i in range(len(nums)):
                if i > 0 and nums[i - 1] == nums[i]:
                    continue
                backtrack(nums[i + 1: ], ans + [nums[i]], curr + nums[i])
        backtrack(candidates, [], 0)
        return res
```

### 41. First Missing Positive

- set

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        s = set(nums)
        for i in range(1, len(nums) + 2):
            if i not in s:
                return i
```

- array index
```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # 3 passes
        n = len(nums)
        for i in range(n): # find negative
            if nums[i] < 0:
                nums[i] = 0
        for i in range(n): # sign positive to negative
            val = abs(nums[i])
            if 1 <= val <= n:
                if nums[val - 1] > 0:
                    nums[val - 1] *= -1
                elif nums[val - 1] == 0: # handle 0, sign to negative out of [1, len(nums)]
                    nums[val - 1] = -1 * (n + 1)
        for i in range(1, n + 1): # find first positive or 0
            if nums[i - 1] >= 0:
                return i
        return len(nums) + 1
```

### 42. Trapping Rain Water

- two pointers

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        l, r, res = 0, len(height) - 1, 0
        leftMax, rightMax = height[l], height[r]
        while l < r:
            if leftMax < rightMax:
                l += 1
                leftMax = max(leftMax, height[l])
                res += leftMax - height[l]
            else:
                r -= 1
                rightMax = max(rightMax, height[r])
                res += rightMax - height[r]
        return res
```

### 43. Multiply Strings

- math

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        def str_to_num(n):
            res = 0
            for i in range(len(n)):
                res = res * 10 + ord(n[i]) - ord('0')
            return res
        res = str_to_num(num1) * str_to_num(num2)
        # convert to string, remember to reverse the result 
        ans = '0' 
        while res:
            ans = (chr(ord('0') + res % 10)) + ans
            res //= 10 
        return ans[:-1] if len(ans) > 1 else ans
```

### 44. Wildcard Matching

- dp

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        R, C = len(s), len(p)
        dp = [[False for j in range(C + 1)] for i in range(R + 1)]
        dp[0][0] = True
        for j in range(1, C + 1):
            if p[j - 1] == "*" and dp[0][j - 1]:
                dp[0][j] = True
        
        for i in range(1, R + 1):
            for j in range(1, C + 1):
                if p[j-1] == s[i-1] or p[j-1] == "?":
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == "*":
                    dp[i][j] = dp[i-1][j] or dp[i][j-1] or dp[i-1][j-1]
        return dp[-1][-1]
```

### 45. Jump Game II

- sliding window

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        res, l, r = 0, 0, 0
        while r < len(nums) - 1:
            furthest = 0
            for i in range(l, r + 1):
                furthest = max(furthest, nums[i] + i)
            res += 1
            l += 1
            r = furthest
        return res
```

### 46. Permutations

- backtrack

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, ans, res):
            if not nums:
                res.append(ans[::])
            for i in range(len(nums)):
                ans.append(nums[i])
                backtrack(nums[:i] + nums[i+1:], ans, res)
                ans.pop()
            return res
        return backtrack(nums, [], [])
```

### 47. Permutations II

- same as: 46

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, ans, s):
            if not nums:
                s.add(tuple(ans[::]))
            for i in range(len(nums)):
                ans.append(nums[i])
                backtrack(nums[:i] + nums[i+1:], ans, s)
                ans.pop()
            return s
        return backtrack(nums, [], set())
```

### 48. Rotate Image

- matrix

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, ans, s):
            if not nums:
                s.add(tuple(ans[::]))
            for i in range(len(nums)):
                ans.append(nums[i])
                backtrack(nums[:i] + nums[i+1:], ans, s)
                ans.pop()
            return s
        return backtrack(nums, [], set())
```

### 49. Group Anagrams

- hash table

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            res = ''.join(sorted(list(s)))
            d[res].append(s)
        return list(d.values())
```

### 50. Pow(x, n)

- binary search

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):
            if x == 0:
                return 0
            if n == 0:
                return 1
            res = helper(x, n // 2)
            res *= res
            return res * x if n % 2 else res
        res = helper(x, abs(n))
        return res if n >= 0 else 1 / res
```
