## template

- https://leetcode.cn/circle/discuss/9oZFK9/

## Question list

### 1 Next Greater or smaller (7)

- next greater

* [739. Daily Temperatures](#739-Daily-Temperatures)
* [496. Next Greater Element I](#496-next-greater-element-i)
* [503. Next Greater Element II](#503-next-greater-element-ii)
* [1019. Next Greater Node In Linked List](#1019-next-greater-node-in-linked-list)

- next smaller or equal

* [1475. Final Prices With a Special Discount in a Shop](#1475-final-prices-with-a-special-discount-in-a-shop)

- next greater or equal

* [901. Online Stock Span](#456-132-Pattern)* 
* [456. 132 Pattern](#456-132-Pattern)

### 2 Longest interval (2)

* [962. Maximum Width Ramp](#962-maximum-width-ramp)
* [1124. Longest Well-Performing Interval](#1124-longest-well-performing-interval)

### 3 Rectangle Max Area (5)

* [42. Trapping Rain Water](#42-trapping-rain-water)
* [84. Largest Rectangle in Histogram](#84-largest-rectangle-in-histogram)
* [85. Maximal Rectangle](#85-maximal-rectangle)
* [1504. Count Submatrices With All Ones](#1504-count-submatrices-with-all-ones)
* [1793. Maximum Score of a Good Subarray](#1793-maximum-score-of-a-good-subarray)

### 4 Contributions (3)

* [907. Sum of Subarray Minimums 1976](#907-sum-of-subarray-minimums)
* [2104. Sum of Subarray Ranges 2000]()
* [1856. Maximum Subarray Min-Product 2051]()

### 5 Dictionary sequence (4)

* [402. Remove K Digits ~1800](#402-remove-k-digits)
* [316. Remove Duplicate Letters](#316-remove-duplicate-letters)
* [1081. Smallest Subsequence of Distinct Characters](#1081-smallest-subsequence-of-distinct-characters)
* [1673. Find the Most Competitive Subsequence 1802](#1673-find-the-most-competitive-subsequence)

### 6 Two monotonic stack (2)

* [2866. Beautiful Towers II 2071](#2866-beautiful-towers-ii)
* [2454. Next Greater Element IV](#2454-next-greater-element-iv)

### 7 Hash + monotonic stack (2)

* [1944. Number of Visible People in a Queue](#1944-number-of-visible-people-in-a-queue)
* [3113. Find the Number of Subarrays Where Boundary Elements Are Maximum](#3113-find-the-number-of-subarrays-where-boundary-elements-are-maximum)

## 8 Dp + monotonic stack (1)

* [2289. Steps to Make Array Non-decreasing](#2289-steps-to-make-array-non-decreasing)

## Exercise list                                                                                            


### 1 Next Greater or smaller (7)

- next greater

* [739. Daily Temperatures](#739-Daily-Temperatures)
* [496. Next Greater Element I](#496-next-greater-element-i)
* [503. Next Greater Element II](#503-next-greater-element-ii)
* [1019. Next Greater Node In Linked List](#1019-next-greater-node-in-linked-list)

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

### 496. Next Greater Element I

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        d, stack = defaultdict(int), []
        for i, n in enumerate(nums2):
            while stack and n > nums2[stack[-1]]:
                d[nums2[stack[-1]]] = n 
                stack.pop()
            stack.append(i)
        return [d.get(n, -1) for n in nums1]
```

### 503. Next Greater Element II

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n, stack = len(nums), []
        nums = nums * 2
        res = [-1] * 2 * n 
        for i, x in enumerate(nums):
            while stack and x > nums[stack[-1]]:
                j = stack.pop()
                res[j] = x
            stack.append(i)
        return res[: n]
```

### 1019. Next Greater Node In Linked List

```python
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        arr = []
        while head:
            arr.append(head.val)
            head = head.next 

        n = len(arr)
        res, stack = [0] * n, []
        for i, x in enumerate(arr):
            while stack and x > arr[stack[-1]]: # next great
                j = stack.pop()
                res[j] = x
            stack.append(i)
        return res
```

- next smaller or equal

* [1475. Final Prices With a Special Discount in a Shop](#1475-final-prices-with-a-special-discount-in-a-shop)

### 1475. Final Prices With a Special Discount in a Shop

```python
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        n = len(prices)
        res, stack = prices[::], []
        for i, x in enumerate(prices):
            while stack and x <= prices[stack[-1]]: # next smaller and equal
                j = stack.pop()
                res[j] = prices[j] - x
            stack.append(i)
        return res
```

- next greater or equal

* [901. Online Stock Span](#901-online-stock-span)
* [456. 132 Pattern](#456-132-Pattern)

### 901. Online Stock Span

```python
class StockSpanner:

    def __init__(self):
        self.stack = []
        self.i = 0

    def next(self, price: int) -> int:
        res, j = 1, self.i
        while self.stack and price >= self.stack[-1][0]:
            j = self.stack.pop()[1]
            res = max(res, self.i - j + 1)
        self.stack.append((price, j))
        self.i += 1
        return res
```

### 456. 132 Pattern

```python
from sortedcontainers import SortedList
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        mn = nums[0]
        sl = SortedList(nums[2:])
        for j in range(1, len(nums) - 1):
            if nums[j] > mn:
                k = sl.bisect_right(mn)
                if k < len(sl) and sl[k] < nums[j]:
                    return True
            mn = min(mn, nums[j])
            sl.remove(nums[j + 1])
        return False
```

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        prefixMin, stack = nums[::], []
        n = len(nums)
        for i in range(1, n):
            prefixMin[i] = min(prefixMin[i], prefixMin[i - 1])
        
        mn = nums[0]
        stack = []
        for k in range(1, n):
            while stack and nums[k] >= nums[stack[-1]]:
                j = stack.pop()
            if stack and prefixMin[stack[-1] - 1] < nums[k]:
                return True
            stack.append(k)
        return False
```

### 2 Longest interval (2)

* [962. Maximum Width Ramp](#962-maximum-width-ramp)
* [1124. Longest Well-Performing Interval](#1124-longest-well-performing-interval)

### 962. Maximum Width Ramp

```python
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        res, n = 0, len(nums)
        stack = [0]
        for i in range(1, n):
            if nums[i] < nums[stack[-1]]:
                stack.append(i)
        for i in range(n - 1, -1, -1):
            while stack and nums[i] >= nums[stack[-1]]:
                j = stack.pop()
                res = max(res, i - j)
        return res 
```

### 1124. Longest Well-Performing Interval

```python
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        n = len(hours)
        prefix = [-1] * n
        for i, h in enumerate(hours):
            if h > 8:
                prefix[i] = 1
        prefix = list(accumulate(prefix, initial = 0))

        stack = [0]
        for i in range(1, len(prefix)):
            if prefix[i] < prefix[stack[-1]]:
                stack.append(i)
        res = 0
        for i in range(n, -1, -1):
            while stack and prefix[i] > prefix[stack[-1]]:
                j = stack.pop()
                res = max(res, i - j)
        return res
```
### 3 Rectangle Max Area (5)

* [42. Trapping Rain Water](#42-trapping-rain-water)
* [84. Largest Rectangle in Histogram](#84-largest-rectangle-in-histogram)
* [85. Maximal Rectangle](#85-maximal-rectangle)
* [1504. Count Submatrices With All Ones](#1504-count-submatrices-with-all-ones)
* [1793. Maximum Score of a Good Subarray](#1793-maximum-score-of-a-good-subarray)

### 42. Trapping Rain Water

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        res, stack = 0, []
        for i, h in enumerate(height):
            while stack and height[stack[-1]] < h:
                j = stack.pop()
                bottom_h = height[j]
                if not stack:
                    break
                l = stack[-1]
                res += (min(height[l], h) - bottom_h) * (i - l - 1)
            stack.append(i)
        return res
```

### 84. Largest Rectangle in Histogram

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        res, stack, heights = 0, [-1], heights + [0]
        for i in range(len(heights)):
            while len(stack) > 1 and heights[stack[-1]] > heights[i]:
                j = stack.pop()
                res = max(res, (i - 1 - stack[-1]) * heights[j])
            stack.append(i)
        return res
```

### 85. Maximal Rectangle

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def fn(heights):
            res, stack, heights = 0, [-1], heights + [0]
            for i in range(len(heights)):
                while len(stack) > 1 and heights[stack[-1]] > heights[i]:
                    j = stack.pop()
                    res = max(res, (i - 1 - stack[-1]) * heights[j])
                stack.append(i)
            return res

        R, C, res = len(matrix), len(matrix[0]), 0
        matrix = [list(map(int, item)) for item in matrix]
        for r in range(R):
            for c in range(C):
                if r > 0 and matrix[r][c]:
                    matrix[r][c] += matrix[r - 1][c]
            res = max(res, fn(matrix[r]))
        return res
```

### 1504. Count Submatrices With All Ones

```python
class Solution:
    def numSubmat(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        dp = [[0] * C for r in range(R)]

        res = 0
        for i in range(R):
            for j in range(C):
                if mat[i][j]:
                    dp[i][j] = dp[i][j-1] + 1 if j > 0 else 1
                    mn = inf
                    for k in range(i, -1, -1):
                        mn = min(mn, dp[k][j]) # row i and row i - 1 to combine
                        if mn == 0: 
                            break
                        res += mn          
        return res
```

```python
class Solution:
    def numSubmat(self, mat: List[List[int]]) -> int:
        R, C, res = len(mat), len(mat[0]), 0
        hist = [0] * (C + 1) # design hist for hist[-1] to be 0
        for i in range(R):
            stack, dp = [-1], [0] * (C + 1) # design of stack for easy calc
            for j in range(C):
                hist[j] = 0 if mat[i][j] == 0 else hist[j] + 1
                while hist[j] < hist[stack[-1]]:
                    stack.pop()
                dp[j] = dp[stack[-1]] + hist[j] * (j - stack[-1]) # Important!!
                stack.append(j)
            res += sum(dp)
        return res
```

### 1793. Maximum Score of a Good Subarray

```python
class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        res, stack, nums = 0, [-1], nums + [0]
        for i in range(len(nums)):
            while len(stack) > 1 and nums[stack[-1]] > nums[i]:
                j = stack.pop()
                if stack[-1] + 1 <= k <= i - 1:
                    res = max(res, (i - 1 - stack[-1]) * nums[j])
            stack.append(i)
        return res
```

### 4 Contributions

* [907. Sum of Subarray Minimums 1976](#907-sum-of-subarray-minimums)
* [2104. Sum of Subarray Ranges 2000](#2104-sum-of-subarray-ranges)
* [1856. Maximum Subarray Min-Product 2051](#1856-maximum-subarray-min-product)

### 907. Sum of Subarray Minimums

```python
'''
contribution template, calculate each contribute of the element and sum of all
of them
'''
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        left, right = [-1] * n, [n] * n
        mod = 10 ** 9 + 7
        stack = []
        for i, x in enumerate(arr):
            while stack and x < arr[stack[-1]]: # previous smaller and equal to avoid duplicate with next smaller
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        stack = []
        for i, x in enumerate(arr):
            while stack and x < arr[stack[-1]]: # next smaller
                j = stack.pop()
                right[j] = i
            stack.append(i)
        return sum((i - left[i]) * (right[i] - i) * v for i, v in enumerate(arr)) % mod
```

### 2104. Sum of Subarray Ranges

```python
'''
By using O(n) would be a hard problem, same as 907. Sum of Subarray Minimums
'''
class Solution:
    def subArrayRanges(self, arr: List[int]) -> int:
        n = len(arr)
        left, right = [-1] * n, [n] * n
        stack = []
        for i, v in enumerate(arr):
            while stack and arr[stack[-1]] > v:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        stack = []
        for i, x in enumerate(arr):
            while stack and x < arr[stack[-1]]: # next smaller
                j = stack.pop()
                right[j] = i
            stack.append(i)
        mn = sum((i - left[i]) * (right[i] - i) * v for i, v in enumerate(arr))

        left, right = [-1] * n, [n] * n
        stack = []
        for i, v in enumerate(arr):
            while stack and arr[stack[-1]] < v:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        stack = []
        for i, x in enumerate(arr):
            while stack and x > arr[stack[-1]]: # next bigger
                j = stack.pop()
                right[j] = i
            stack.append(i)
        mx = sum((i - left[i]) * (right[i] - i) * v for i, v in enumerate(arr))
        return mx - mn
```

### 1856. Maximum Subarray Min-Product

```python
class Solution:
    def maxSumMinProduct(self, arr: List[int]) -> int:
        pre = list(accumulate(arr, initial = 0))
        n = len(arr)
        left, right = [-1] * n, [n] * n
        mod = 10 ** 9 + 7
        stack = []
        for i, x in enumerate(arr):
            while stack and x <= arr[stack[-1]]:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        stack = []
        for i, x in enumerate(arr):
            while stack and x < arr[stack[-1]]:
                j = stack.pop()
                right[j] = i
            stack.append(i)
        return max(arr[i - 1] * (pre[right[i - 1]] - pre[left[i - 1] + 1]) for i in range(1, len(pre))) % mod
```

### 5 Dictionary sequence

* [402. Remove K Digits ~1800](#402-remove-k-digits)
* [316. Remove Duplicate Letters](#316-remove-duplicate-letters)
* [1081. Smallest Subsequence of Distinct Characters](#1081-smallest-subsequence-of-distinct-characters)
* [1673. Find the Most Competitive Subsequence 1802](#1673-find-the-most-competitive-subsequence)

### 402. Remove K Digits

```python
'''
    template for dictionary sequence by using monotonic stack
'''
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack, remain = [], len(num) - k
        for n in num:
            while k and stack and n < stack[-1]:
                k -= 1
                stack.pop()
            stack.append(n)
        return ''.join(stack)[: remain].lstrip('0') or '0'
```

### 316. Remove Duplicate Letters

```python
'''
    check 402. Remove K Digits
'''
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack, remain, visited = [], Counter(s), set()
        for c in s:
            if c not in visited:
                while stack and c < stack[-1] and remain[stack[-1]] > 0:
                    visited.remove(stack.pop())
                stack.append(c)
                visited.add(c)
            remain[c] -= 1
        return ''.join(stack)
```

### 1081. Smallest Subsequence of Distinct Characters

```python
'''
    same as 316. Remove Duplicate Letters
'''
class Solution:
    def smallestSubsequence(self, s: str) -> str:
        stack, remain, visited = [], Counter(s), set()
        for c in s:
            if c not in visited:
                while stack and c < stack[-1] and remain[stack[-1]] > 0:
                    visited.remove(stack.pop())
                stack.append(c)
                visited.add(c)
            remain[c] -= 1
        return ''.join(stack)
```

### 1673. Find the Most Competitive Subsequence

```python
'''
same as 402. Remove K Digits
'''
class Solution:
    def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
        stack, remove = [], len(nums) - k 
        for n in nums:
            while remove and stack and n < stack[-1]:
                remove -= 1
                stack.pop()
            stack.append(n)
        return stack[:k]
```

### 6 Two monotonic stack (2)

* [2832. Maximal Range That Each Element Is Maximum in It](#2832-maximal-range-that-each-element-is-maximum-in-it)
* [2866. Beautiful Towers II 2071](#2866-beautiful-towers-ii)
* [2454. Next Greater Element IV](#2454-next-greater-element-iv)

### 2832. Maximal Range That Each Element Is Maximum in It

```python
class Solution:
    def maximumLengthOfRanges(self, nums: List[int]) -> List[int]:
        stack = []
        nums = [inf] + nums + [inf]
        n = len(nums)
        res = [1] * n 
        for i, x in enumerate(nums):
            while stack and x > nums[stack[-1]]:
                j = stack.pop()
                res[j] += i - j - 1
            stack.append(i)
        
        stack = []
        nums = nums[::-1]
        for i, x in enumerate(nums):
            while stack and x > nums[stack[-1]]:
                j = stack.pop()
                res[n - j - 1] += i - j - 1
            stack.append(i)
        return res[1:-1]
```

### 2866. Beautiful Towers II

```python
'''
    monotonic increasing stack from left to right
    monotonic increasing stack from right to left
'''
class Solution:
    def maximumSumOfHeights(self, a: List[int]) -> int:
        n = len(a)
        st = [n]
        s = 0
        suf = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            x = a[i]
            while len(st) > 1 and x <= a[st[-1]]:
                j = st.pop()
                s -= a[j] * (st[-1] - j)
            s += x * (st[-1] - i)
            suf[i] = s
            st.append(i)
        
        st = [-1]
        s = 0
        pre = [0] * n
        for i, x in enumerate(a):
            while len(st) > 1 and x <= a[st[-1]]:
                j = st.pop()
                s -= a[j] * (j - st[-1])
            s += x * (i - st[-1])
            pre[i] = s
            st.append(i)
        
        res = suf[0]
        for i in range(n):
            res = max(res, pre[i] + suf[i + 1])
        return res
```

```python
class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        def leftMax(A):
            left, stack, total = [0] * n, [-1], 0
            for i in range(n):
                while len(stack) > 1 and A[stack[-1]] > A[i]:
                    j = stack.pop()
                    total -= (j - stack[-1]) * A[j]
                total += (i - stack[-1]) * A[i]
                stack.append(i)
                left[i] = total
            return left

        left, right = leftMax(maxHeights), leftMax(maxHeights[::-1])[::-1]
        return max(l + r - v for l, r, v in zip(left, right, maxHeights))
```

### 2454. Next Greater Element IV

```python
class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [-1] * n 
        stack1 = []  
        stack2 = []
        for i, x in enumerate(nums):
            while stack2 and x > nums[stack2[-1]]:
                res[stack2.pop()] = x 
            q = deque()
            while stack1 and x > nums[stack1[-1]]:
                j = stack1.pop()
                q.appendleft(j)
            stack2 += q
            stack1.append(i)
        return res 
```

### 7 array + monotonic stack (2)

* [1944. Number of Visible People in a Queue](#1944-number-of-visible-people-in-a-queue)
* [3113. Find the Number of Subarrays Where Boundary Elements Are Maximum](#3113-find-the-number-of-subarrays-where-boundary-elements-are-maximum)

### 1944. Number of Visible People in a Queue

```python
class Solution:
    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        stack = []
        n = len(heights)
        res = [0] * n 
        for i, x in enumerate(heights):
            while stack and x >= heights[stack[-1]]:
                j = stack.pop()
                res[j] += 1
            if stack:
                res[stack[-1]] += 1
            stack.append(i)
        return res 
```

### 3113. Find the Number of Subarrays Where Boundary Elements Are Maximum

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        d = {i: 1 for i in range(len(nums))}
        stack = []
        for i, x in enumerate(nums):
            while stack and x >= nums[stack[-1]]:
                j = stack.pop()
                if nums[j] == nums[i]:
                    d[i] += d[j]
            stack.append(i)
        return sum(d.values())
```

## 8 Dp + monotonic stack (1)

* [2289. Steps to Make Array Non-decreasing](#2289-steps-to-make-array-non-decreasing)

### 2289. Steps to Make Array Non-decreasing

```python
class Solution:
    def totalSteps(self, nums: List[int]) -> int:
        res, stack = 0, [] # (time, i)
        for i, x in enumerate(nums):
            t = 0
            while stack and x >= nums[stack[-1][1]]:
                t = max(t, stack.pop()[0])
            if stack:
                t += 1
            res = max(res, t)
            stack.append((t, i))
        return res 
```



## left

### 1776. Car Fleet II

### 1966. Binary Searchable Numbers in an Unsorted Array

### 2818. Apply Operations to Maximize Score

### 2281. Sum of Total Strength of Wizards

### 321. Create Maximum Number

```python
lass Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def divide(nums, k):
            stack = []
            drop = len(nums) - k
            for num in nums:
                while drop and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:k]

        def merge(A, B):
            ans = []
            while A or B:
                bigger = A if A > B else B
                ans.append(bigger.pop(0))
            return ans
        
        res = [0] * k
        for i in range(k + 1):
            if i <= len(nums1) and k-i <= len(nums2):
                res = max(res, merge(divide(nums1, i), divide(nums2, k-i)))
        return res
```

### 2030. Smallest K-Length Subsequence With Occurrences of a Letter

```python
```