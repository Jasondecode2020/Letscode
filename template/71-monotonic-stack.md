### 907. Sum of Subarray Minimums

```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        left, right = [-1] * n, [n] * n
        mod = 10 ** 9 + 7
        stack = []
        for i, v in enumerate(arr):
            while stack and arr[stack[-1]] > v:
                stack.pop()
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] >= arr[i]:
                stack.pop()
            if stack:
                right[i] = stack[-1]
            stack.append(i)
        return sum((i - left[i]) * (right[i] - i) * v for i, v in enumerate(arr)) % mod
```

### 2104. Sum of Subarray Ranges

```python
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        n, res = len(nums), 0
        for i in range(n):
            mn, mx = nums[i], nums[i]
            for j in range(i, n):
                mn = min(mn, nums[j])
                mx = max(mx, nums[j])
                res += mx - mn
        return res

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
        for i in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] >= arr[i]:
                stack.pop()
            if stack:
                right[i] = stack[-1]
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
        for i in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] <= arr[i]:
                stack.pop()
            if stack:
                right[i] = stack[-1]
            stack.append(i)

        mx = sum((i - left[i]) * (right[i] - i) * v for i, v in enumerate(arr))
        return mx - mn
```