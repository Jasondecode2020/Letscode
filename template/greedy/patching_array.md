- same idea of patching array

* [1798. Maximum Number of Consecutive Values You Can Make](#1798-maximum-number-of-consecutive-values-you-can-make)
* [330. Patching Array](#330-patching-array)
* [2952. Minimum Number of Coins to be Added](#2952-minimum-number-of-coins-to-be-added)

### 1798. Maximum Number of Consecutive Values You Can Make

```python 
class Solution:
    def getMaximumConsecutive(self, coins: List[int]) -> int:
        coins.sort()
        s = 1
        i = 0
        res = 0
        mx = sum(coins)
        while s <= mx:
            if i < len(coins) and coins[i] <= s:
                s += coins[i]
                i += 1
            else:
                break 
        return s 
```

### 330. Patching Array

```python 
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        nums.sort()
        i = 0
        s = 1
        res = 0
        while s <= n:
            if i < len(nums) and nums[i] <= s:
                s += nums[i]
                i += 1
            else:
                s += s 
                res += 1
        return res
```

### 2952. Minimum Number of Coins to be Added

```python 
class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        i = 0
        s = 1
        res = 0
        while s <= target:
            if i < len(coins) and coins[i] <= s:
                s += coins[i]
                i += 1
            else:
                s += s 
                res += 1
        return res 
```