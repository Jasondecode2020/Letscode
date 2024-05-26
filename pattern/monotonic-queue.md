### 1 monototic decreasing queue

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

### 2 monotonic increasing queue

```python
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