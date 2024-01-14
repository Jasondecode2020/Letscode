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