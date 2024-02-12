## hash(2)

* [1. Two Sum](#1-Two-Sum)
* [560. Subarray Sum Equals K](#560-Subarray-Sum-Equals-K)


### 1. Two Sum

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

### 560. Subarray Sum Equals K

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res, presum, d = 0, 0, {0: 1} # [1, 2, 3]
        for n in nums:
            presum += n
            if presum - k in d:
                res += d[presum - k]
            d[presum] = d.get(presum, 0) + 1
        return res
```
