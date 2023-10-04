## template

```python
def kmp(s):
    nxt, j = [-1], -1
    for i in range(len(s)):
        while j >= 0 and s[i] != s[j]:
            j = nxt[j]
        j += 1
        nxt.append(j)
    return nxt # nxt[i]: i - 1结尾的最大真前缀长度
```

### 1392. Longest Happy Prefix

```python
class Solution:
    def longestPrefix(self, s: str) -> str:
        def kmp(s):
            nxt, j = [-1], -1
            for i in range(len(s)):
                while j >= 0 and s[i] != s[j]:
                    j = nxt[j]
                j += 1
                nxt.append(j)
            return nxt

        k = kmp(s)[-1]
        return s[: k]
```