### 1310. XOR Queries of a Subarray

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        arr = [0] + arr
        for i in range(1, len(arr)):
            arr[i] ^= arr[i - 1]
        res = []
        for s, e in queries:
            res.append(arr[e + 1] ^ arr[s])
        return res
```

### 1177. Can Make Palindrome from Substring

```python
class Solution:
    def canMakePaliQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        dp = [[0] * 26 for i in range(n + 1)]
        for i, c in enumerate(s, start = 1):
            dp[i][ord(c) - ord('a')] += 1
        for i in range(1, n + 1):
            for j in range(26):
                dp[i][j] += dp[i - 1][j]

        res = [False] * len(queries)
        for j, (l, r, k) in enumerate(queries):
            odd = 0
            for i in range(26):
                if (dp[r + 1][i] - dp[l][i]) % 2 == 1:
                    odd += 1
            if odd // 2 <= k:
                res[j] = True
        return res
```

### 1371. Find the Longest Substring Containing Vowels in Even Counts

```python
class Solution:
    def findTheLongestSubstring(self, s: str) -> int:
        idx = [-1] + [inf] * ((1 << 5) - 1)
        res, mask = 0, 0
        vowel = set(list('aeiou'))
        d = Counter()
        for i, c in enumerate('aeoui'):
            d[c] = i
        for i, c in enumerate(s):
            if c in vowel:
                mask ^= 1 << d[c]
            res = max(res, i - idx[mask])
            idx[mask] = min(i, idx[mask])
        return res 
```

### 1915. Number of Wonderful Substrings

```python
class Solution:
    def wonderfulSubstrings(self, word: str) -> int:
        index = [1] + [0] * ((1 << 10) - 1)
        res, mask = 0, 0
        for i, c in enumerate(word):
            mask ^= 1 << (ord(c) - ord('a'))
            res += index[mask]
            for j in range(10):
                res += index[mask ^ (1 << j)]
            index[mask] += 1
        return res
```