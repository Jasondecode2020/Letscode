# Hash Table

This file collects five hash-table problems and solutions selected from the existing `pattern` and `template` folders.

## 1. Two Sum
Problem: Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, v in enumerate(nums):
            need = target - v
            if need in d:
                return [d[need], i]
            d[v] = i
        return []
```

## 219. Contains Duplicate II
Problem: Given an integer array `nums` and an integer `k`, return `true` if there are two distinct indices `i` and `j` such that `nums[i] == nums[j]` and `abs(i - j) <= k`.

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        last_index = {}
        for i, n in enumerate(nums):
            if n in last_index and i - last_index[n] <= k:
                return True
            last_index[n] = i
        return False
```

## 1128. Number of Equivalent Domino Pairs
Problem: Given a list of domino pairs, count the number of pairs `(i, j)` such that the dominoes are equivalent.

```python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        counts = {}
        res = 0
        for a, b in dominoes:
            if a > b:
                a, b = b, a
            key = (a, b)
            res += counts.get(key, 0)
            counts[key] = counts.get(key, 0) + 1
        return res
```

## 2441. Largest Positive Integer That Exists With Its Negative
Problem: Given an integer array `nums`, return the largest positive integer `k` such that both `k` and `-k` exist in `nums`. If no such integer exists, return `-1`.

```python
class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        seen = set()
        res = -1
        for n in nums:
            if -n in seen:
                res = max(res, abs(n))
            seen.add(n)
        return res
```

## 1461. Check If a String Contains All Binary Codes of Size K
Problem: Given a binary string `s` and an integer `k`, return `true` if every binary code of length `k` exists as a substring of `s`.

```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        if len(s) < k:
            return False

        codes = set()
        for i in range(len(s) - k + 1):
            codes.add(s[i:i+k])
        return len(codes) == 1 << k
```
