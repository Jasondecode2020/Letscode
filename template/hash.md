## Hash

### 336. Palindrome Pairs

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        lookup = {w: i for i, w in enumerate(words)}
        res = []
        for i, w in enumerate(words):
            for j in range(len(w) + 1):
                pre, suf = w[:j], w[j:]
                if pre[::-1] == pre and suf[::-1] != w and suf[::-1] in lookup:
                    res.append([lookup[suf[::-1]], i])
                if suf[::-1] == suf and pre[::-1] != w and pre[::-1] in lookup and j != len(w):
                    # j != len(w)，j = w的情况已经出现过, avoid duplicate
                    res.append([i, lookup[pre[::-1]]])
        return res
```

### 594. Longest Harmonious Subsequence

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        d = Counter(nums)
        res = 0
        for k in d:
            if k - 1 in d:
                res = max(res, d[k] + d[k - 1])
            if k + 1 in d:
                res = max(res, d[k] + d[k + 1])
        return res
```

### 1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers

```python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        def check(nums1, nums2):
            c1, c2 = Counter(nums1), Counter(nums2)
            res = 0
            for n1 in c1:
                for n2 in c2:
                    if (n1 * n1) % n2 == 0:
                        if n1 * n1 // n2 in c2 and n1 == n2:
                            res += c1[n1] * c2[n2] * (c2[n2] - 1)
                        if n1 * n1 // n2 in c2 and n1 != n2:
                            res += c1[n1] * c2[n2] * c2[n1 * n1 // n2]
            return res // 2
        return check(nums1, nums2) + check(nums2, nums1)
```

### 1647. Minimum Deletions to Make Character Frequencies Unique

```python
class Solution:
    def minDeletions(self, s: str) -> int:
        c = Counter(s)
        seen = set()
        res = 0
        for v in sorted(c.values()):
            if v not in seen:
                seen.add(v)
            else:
                while v in seen and v - 1 >= 0:
                    v -= 1
                    res += 1
                seen.add(v)
        return res
```