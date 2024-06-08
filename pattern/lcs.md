## template

```python
def lcs(s1, s2):
    R, C = len(s1) + 1, len(s2) + 1
    f = [[0] * C for i in range(R)]
    for i in range(1, R):
        for j in range(1, C):
            if s1[i - 1] == s2[j - 1]:
                f[i][j] = 1 + f[i - 1][j - 1]
            else:
                f[i][j] = max(f[i][j - 1], f[i - 1][j])
    return f[-1][-1]
```