## rolling hash

```python
```

### 1044. Longest Duplicate Substring

```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        mod = 2 ** 32 + 1
        mod2 = 2 ** 32 - 1
        P = 26
        S = [ord(c) - ord('a') for c in s]
        def check(threshold):
            L = threshold
            h = 0
            h2 = 0
            PP = (P ** L) % mod
            PP2 = (P ** L) % mod2
            # init the window
            # abcde
            for i, n in enumerate(S[:L]):
                h = (h * P + n) % mod
                h2 = (h2 * P + n) % mod2
            visited = set([(h, h2)])
            # sliding window
            for r, n in enumerate(S[L:], L):
                h = (h * P - S[r - L] * PP + n) % mod
                h2 = (h2 * P - S[r - L] * PP2 + n) % mod2
                if (h, h2) in visited:
                    return r - L + 1, L
                visited.add((h, h2))
            return False 

        l, r, res = 0, len(s) - 1, 0
        while l <= r:
            m = l + (r - l) // 2
            if check(m):
                res = check(m)
                l = m + 1
            else:
                r = m - 1
        idx, L = res
        return s[idx: idx + L]
```