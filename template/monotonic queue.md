## template: monototic queue

- monototic decreasing queue

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q, res = deque(), []
        for i, n in enumerate(nums):
            # pop and append as monotonic stack
            while q and nums[q[-1]] < n:
                q.pop()
            q.append(i)
            # check sliding window
            if i - q[0] + 1 > k:
                q.popleft()
            # get res
            if i >= k - 1:
                res.append(nums[q[0]])
        return res
```

### 239. Sliding Window Maximum

```python
'''
nlogn: SortedList
'''
from sortedcontainers import SortedList
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        sl = SortedList()
        l, res = 0, []
        for r, n in enumerate(nums):
            sl.add(n)
            if len(sl) == k:
                res.append(sl[-1])
                sl.remove(nums[l])
                l += 1
        return res
```

```python
'''
monotonic decreasing queue
'''
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res, q = [], deque()
        for r, n in enumerate(nums):
            while q and n > nums[q[-1]]:
                q.pop()
            q.append(r)
            if r - q[0] + 1 > k:
                q.popleft()
            if r + 1 >= k:
                res.append(nums[q[0]])
        return res
```

### 862. Shortest Subarray with Sum at Least K

```python
'''
monotonic increasing queue
'''
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        pre = list(accumulate(nums, initial = 0))
        q, res = deque(), inf
        for r, n in enumerate(pre):
            while q and n < pre[q[-1]]:
                q.pop()
            q.append(r)
            while q and pre[r] - pre[q[0]] >= k:
                res = min(res, r - q[0])
                q.popleft()
        return res if res != inf else -1
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
```

### 907. Sum of Subarray Minimums


```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        left, right = [-1] * n, [n] * n
        mod = 10 ** 9 + 7
        # [1, 2, 4]
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

### 2398. Maximum Number of Robots Within Budget

```python
from sortedcontainers import SortedList
class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        n = len(chargeTimes)
        l, res, sl, total = 0, 0, SortedList(), 0
        for r, c in enumerate(chargeTimes):
            sl.add(c)
            total += runningCosts[r]
            L = r - l + 1
            while sl and sl[-1] + L * total > budget:
                sl.remove(chargeTimes[l])
                total -= runningCosts[l]
                l += 1
            res = max(res, r - l + 1)
        return res

# monotonic queue

class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        res = s = l = 0
        q = deque()
        for r, (t, c) in enumerate(zip(chargeTimes, runningCosts)):
            while q and t >= chargeTimes[q[-1]]:
                q.pop()
            q.append(r)
            s += c
            while q and chargeTimes[q[0]] + (r - l  + 1) * s > budget:
                if q[0] == l:
                    q.popleft()
                s -= runningCosts[l]
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 1425. Constrained Subsequence Sum

```python
'''
sorted list
'''
from sortedcontainers import SortedList
class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        sl, l, n = SortedList([nums[0]]), 0, len(nums)
        dp = [nums[0]] * n 
        for r in range(1, n):
            dp[r] = max(nums[r], sl[-1] + nums[r])
            sl.add(dp[r])
            if r - l + 1 > k:
                sl.remove(dp[l])
                l += 1
        return max(dp)
```

```python
'''
monotonic decreasing queue
'''
class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [nums[0]] * n 
        q = deque([0])
        for r in range(1, n):
            dp[r] = max(nums[r], nums[r] + dp[q[0]])
            while q and dp[r] > dp[q[-1]]:
                q.pop()
            q.append(r)
            if r - q[0] + 1 > k:
                q.popleft()
        return max(dp)
```

### 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        l, res, mono_inc_q, mono_dec_q = 0, 0, deque(), deque()
        for r in range(len(nums)): # 10, 11
            while mono_dec_q and mono_dec_q[-1] < nums[r]:
                mono_dec_q.pop()
            mono_dec_q.append(nums[r])
            while mono_inc_q and mono_inc_q[-1] > nums[r]:
                mono_inc_q.pop()
            mono_inc_q.append(nums[r])
            while mono_dec_q[0] - mono_inc_q[0] > limit:
                if nums[l] == mono_dec_q[0]:
                    mono_dec_q.popleft()
                if nums[l] == mono_inc_q[0]:
                    mono_inc_q.popleft()
                l += 1
            res = max(res, r - l + 1)
        return res
```