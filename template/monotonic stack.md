## template

```python
def fn(arr):
    stack, res = [], 0
    for n in arr:
        # for decreasing change '>' to '<'
        while stack and stack[-1] > n:
            # some code
            stack.pop()
        stack.append(n)
    return res
```

### 739. Daily Temperatures

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res = [0] * n
        stack = []
        for i, t in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < t:
                j = stack.pop() 
                res[j] = i - j
            stack.append(i)
        return res
```

### 496. Next Greater Element I

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        d, stack = defaultdict(int), [nums2[0]]
        for i in range(len(nums2)):
            while stack and stack[-1] < nums2[i]:
                d[stack[-1]] = nums2[i]
                stack.pop()
            stack.append(nums2[i])

        res = []
        for n in nums1:
            if n in d:
                res.append(d[n])
            else:
                res.append(-1)
        return res
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

### 316. Remove Duplicate Letters

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # acbd | abcd
        left = Counter(s)
        res = []
        res_set = set()
        for c in s:
            left[c] -= 1
            if c in res_set:
                continue
            while res and c < res[-1] and left[res[-1]]:
                res_set.remove(res.pop())
            res.append(c)
            res_set.add(c)
        return ''.join(res)
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

### 503. Next Greater Element II

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        origin = nums[::]
        nums = nums * 2
        d, stack = defaultdict(list), [origin[0]]
        for i in range(1, len(nums)):
            while stack and stack[-1] < nums[i]:
                d[stack[-1]].append(nums[i])
                stack.pop()
            stack.append(nums[i])
    
        res = []
        for n in origin:
            if n in d:
                res.append(d[n].pop(0))
            else:
                res.append(-1)
        return res
```