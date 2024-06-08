## template 1: O(n^2)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1] * n 
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    f[i] = max(f[i], f[j] + 1)
        return max(f)
```

## template 2: O(nlog(n))

```python
def lis(arr): # more than or equal
    f = []
    for n in arr:
        i = bisect_right(f, n)
        if i == len(f):
            f.append(n)
        else:
            f[i] = n
    return len(f)   
```

```python
def lis(arr): # strictly increasing
    f = []
    for n in arr:
        i = bisect_left(f, n)
        if i == len(Lf):
            f.append(n)
        else:
            f[i] = n
    return len(f)
```

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        def LIS(nums): # space O(1)
            end = 0
            for n in nums:
                i = bisect_left(nums, n, 0, end)
                if i == end:
                    nums[end] = n
                    end += 1
                else:
                    nums[i] = n
            return end
        return LIS(nums) >= 3
```