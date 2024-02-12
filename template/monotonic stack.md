## template

### next greater

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x > arr[stack[-1]]: # next great
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### next greater and equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x >= arr[stack[-1]]: # next great and equal
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### next smaller

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x < arr[stack[-1]]: # next smaller
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### next smaller and equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x <= arr[stack[-1]]: # next smaller and equal
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```
### previous greater

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] <= v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous greater index
        stack.append(i)
```

### previous greater and equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] < v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous greater and equal index
        stack.append(i)
```

### previous smaller

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] >= v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous smaller index
        stack.append(i)
```

### previous smaller and equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] > v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous smaller and equal index
        stack.append(i)
```

### Question list

- next greater
* [3. Longest Substring Without Repeating Characters](#3-Longest-Substring-Without-Repeating-Characters)
* [739. Daily Temperatures](#739-Daily-Temperatures)
* [496. Next Greater Element I](#739-Daily-Temperatures)
* [503. Next Greater Element II](#739-Daily-Temperatures)
* [1019. Next Greater Node In Linked List](#739-Daily-Temperatures)

- next smaller and equal

* [1475. Final Prices With a Special Discount in a Shop](#739-Daily-Temperatures)
* [456. 132 Pattern ~2000 (with prefix sum)](#456-132-Pattern)

- last greater or equal: check from right to left

* [962. Maximum Width Ramp](#456-132-Pattern)
* [1124. Longest Well-Performing Interval](#456-132-Pattern)

- first smaller or equal: check from left to right

* [901. Online Stock Span](#456-132-Pattern)

- two monotonic stack
* [2866. Beautiful Towers II 2071](#456-132-Pattern)

2454. 下一个更大元素 IV 2175
2289. 使数组按非递减顺序排列 2482
1776. 车队 II 2531
2832. 每个元素为最大值的最大范围（会员题）

- rectangle monotonic stack

* [84. Largest Rectangle in Histogram](#456-132-Pattern)
* [85. Maximal Rectangle](#456-132-Pattern)
* [1504. Count Submatrices With All Ones](#456-132-Pattern)

- dictionary monotonic stack

* [402. Remove K Digits ~1800](#456-132-Pattern)
* [316. Remove Duplicate Letters](#456-132-Pattern)
* [1081. Smallest Subsequence of Distinct Characters](#456-132-Pattern)
* [1673. Find the Most Competitive Subsequence 1802](#456-132-Pattern)
321. 拼接最大数


- contributions monotonic stack

* [907. Sum of Subarray Minimums 1976](#456-132-Pattern)
* [2104. Sum of Subarray Ranges 2000](#456-132-Pattern)
* [1856. Maximum Subarray Min-Product 2051](#456-132-Pattern)

2818. 操作使得分最大 2397
2281. 巫师的总力量和（最小值×和） 2621


### 739. Daily Temperatures

```python
'''
    it need to calculate next great element, it monotonic decreasing
    t > temperatures[stack[-1]], means current more than top of stack,
    it is monotonic decreasing stack, 
    otherwise, it is monotonic increasing stack
'''
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
'''
    nearly the same as 739. Daily Temperatures,
    just need to have a hash table to record the result
'''
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
'''
    nearly the same as 496. Next Greater Element I,
    just need to enlarge the nums because of cycle next great element
'''
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n, stack, d = len(nums), [], defaultdict(int)
        nums = nums * 2
        for i, x in enumerate(nums):
            while stack and x > nums[stack[-1]]:
                j = stack.pop()
                d[j] = x 
            stack.append(i)
        return [d.get(i, -1) for i in range(n)]
```


### 1019. Next Greater Node In Linked List

```python
'''
    nearly the same as 739. Daily Temperatures,
    just need to convert linked list to array
'''
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

### 1475. Final Prices With a Special Discount in a Shop

```python
'''
    nearly the same as 739. Daily Temperatures,
    just need to have change to next smaller and equal
'''
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

### 962. Maximum Width Ramp

```python
'''
    more difficult than 739. Daily Temperatures,
    just need to have a monotonic decreasing stack first, used greedy idea,
    to check array from right to left.
'''
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        n, res, stack = len(nums), 0, []
        for i in range(n):
            # keep mono decreasing stack
            if not stack or nums[i] < nums[stack[-1]]:
                stack.append(i)
    
        for i in range(n - 1, -1, -1):
            while stack and nums[i] >= nums[stack[-1]]:
                j = stack.pop()
                res = max(res, i - j)
        return res
```

### 1124. Longest Well-Performing Interval

```python
'''
    more difficult than 962. Maximum Width Ramp,
    just need to convert to 962 first, 
    then have a monotonic decreasing stack first, 
    used greedy idea to check array from right to left.
'''
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        n = len(hours)
        s, stack = [0] * (n + 1), [0] 
        for j, h in enumerate(hours, 1):
            s[j] = s[j - 1] + (1 if h > 8 else -1)
            if s[j] < s[stack[-1]]:
                stack.append(j)

        res = 0
        for i in range(n, 0, -1):
            while stack and s[i] > s[stack[-1]]:
                j = stack.pop()
                res = max(res, i - j)
        return res
```

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
'''
    use sorted list to find a number k,
    check k less j, because k always more than i
'''
from sortedcontainers import SortedList
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        sl = SortedList(nums[2:])
        mn = nums[0]
        for i in range(1, len(nums) - 1):
            if nums[i] > mn:
                j = sl.bisect_right(mn)
                if j < len(sl) and sl[j] < nums[i]:
                    return True
            mn = min(mn, nums[i])
            sl.remove(nums[i + 1])
        return False
```

```python
'''
    monotonic decreasing stack
    and prefix to fix leftmin
'''
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        prefix, stack = nums[::], []
        n = len(nums)
        for i in range(1, n):
            prefix[i] = min(prefix[i], prefix[i - 1])
            
        K = nums.index(prefix[n - 1])
        for j in range(n - 1, 0, -1):
            k = K
            while stack and nums[stack[-1]] < nums[j]:
                k = stack.pop()
            if prefix[j - 1] < nums[k]:
                return True
            stack.append(j)
        return False
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

### 2866. Beautiful Towers II

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

### 84. Largest Rectangle in Histogram

```python
'''
    monotonic increasing stack from left to right
    need to know how to find the max res
'''
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
'''
    similar to 84. Largest Rectangle in Histogram
'''
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
'''
    84. Largest Rectangle in Histogram
'''
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