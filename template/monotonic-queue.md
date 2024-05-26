## Question list

### Monototic queue

* [239. Sliding Window Maximum](#239-sliding-window-maximum)
* [1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](#1438-longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit)
* [2398. Maximum Number of Robots Within Budget](#2398-maximum-number-of-robots-within-budget)
* [862. Shortest Subarray with Sum at Least K](#862-shortest-subarray-with-sum-at-least-k)
* [1499. Max Value of Equation](#1499-max-value-of-equation)
* [2071. Maximum Number of Tasks You Can Assign]()

### 2 dp + monotonic queue

* [1425. Constrained Subsequence Sum](#1425-constrained-subsequence-sum)
* [2944. Minimum Number of Coins for Fruits](#2944-minimum-number-of-coins-for-fruits)
* [1696. Jump Game VI](#1696-jump-game-vi)
* [375. Guess Number Higher or Lower II](#239-sliding-window-maximum)
* [1425. Constrained Subsequence Sum](#239-sliding-window-maximum)




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


### 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

```python
from sortedcontainers import SortedList
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        res = 0
        sl = SortedList()
        l = 0
        for i, x in enumerate(nums):
            sl.add(x)
            while sl and sl[-1] - sl[0] > limit:
                sl.remove(nums[l])
                l += 1
            res = max(res, len(sl))
        return res
```

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

### 1499. Max Value of Equation

```python
class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        # yi + yj + abs(xi - xj), i < j
        # yi + yj + xj - xi
        # yj + xj + yi - xi
        # xj - xi <= k
        # xj <= xi + k 
        res, q = -inf, deque() # (xi, yi - xi)
        for x, y in points:
            while q and x > q[0][0] + k:
                q.popleft()
            if q:
                res = max(res, x + y + q[0][1])
            while q and y - x > q[-1][1]:
                q.pop()
            q.append((x, y - x))
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

### 2944. Minimum Number of Coins for Fruits

```python
class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i):
            if i >= n + 1:
                return 0
            return min(dfs(j) for j in range(i + 1, 2 * i + 2)) + prices[i - 1]
        return dfs(1)
```

### 1696. Jump Game VI

```python
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        @cache
        def dfs(i):
            if i == n - 1:
                return nums[n - 1]
            res = -inf
            for j in range(i + 1, min(n, i + k + 1)):
                res = max(res, dfs(j) + nums[i])
            return res
        return dfs(0)

from sortedcontainers import SortedList
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * n 
        f[0] = nums[0]
        sl = SortedList([nums[0]])
        l = 0
        for i in range(1, n):
            f[i] = sl[-1] + nums[i]
            sl.add(f[i])
            if len(sl) > k:
                sl.remove(f[l])
                l += 1
        return f[-1]

class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * n 
        f[0] = nums[0]
        q = deque([0])
        for i in range(1, n):
            if i - q[0] > k:
                q.popleft()
            f[i] = f[q[0]] + nums[i]
            while q and f[i] > f[q[-1]]:
                q.pop()
            q.append(i)
        return f[-1]
```

### 375. Guess Number Higher or Lower II

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        @cache
        def dfs(i, j):
            if i >= j:
                return 0
            res = inf
            for k in range(i, j + 1):
                res = min(res, max(dfs(i, k - 1), dfs(k + 1, j)) + k)
            return res
        return dfs(1, n)
```