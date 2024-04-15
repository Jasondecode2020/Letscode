### 68. Text Justification

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res, line, i, width = [], [], 0, 0
        while i < len(words):
            word = words[i]
            if width + len(word) <= maxWidth:
                line.append(word)
                width += len(word) + 1
                i += 1
            else:
                spaces = maxWidth - width + len(line)
                space, j = 0, 0
                while space < spaces:
                    if j >= len(line) - 1:
                        j = 0
                    line[j] += ' '
                    space, j = space + 1, j + 1
                res.append(''.join(line))
                line, width = [], 0 
        # last line
        for i in range(len(line) - 1):
            line[i] += ' '
        line[-1] += ' ' * (maxWidth - width + 1)
        res.append(''.join(line))
        return res
```

### 1576. Replace All ?'s to Avoid Consecutive Repeating Characters

```python
class Solution:
    def modifyString(self, s: str) -> str:
        s = '#' + s + '#'
        a = list(s)
        for i in range(1, len(a) - 1):
            if s[i] == '?':
                for c in ascii_lowercase:
                    if c not in a[i - 1] + a[i + 1]:
                        a[i] = c
                        break
        return ''.join(a[1:-1])
```

### 8. String to Integer (atoi)

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        sign = ['+', '-']
        res = ''
        for i, c in enumerate(s):
            if i == 0 and c in sign:
                res += c
                continue
            if c.isdigit():
                res += c 
            else:
                break
        if not res or res in sign:
            return 0
        n = int(res)
        return max(-2 ** 31, min(n, 2 ** 31 - 1))
```

### 831. Masking Personal Information

```python
class Solution:
    def maskPII(self, s: str) -> str:
        if '@' in s:
            s = s.lower()
            s1, s2 = s.split('@')
            return s1[0] + '*****' + s1[-1] + '@' + s2 
        else:
            res = ''.join([c for c in s if c.isdigit()])
            cur = ''
            if len(res) == 10:
                cur = "***-***-"
            elif len(res) == 11:
                cur = "+*-***-***-"
            elif len(res) == 12:
                cur = "+**-***-***-"
            else:
                cur = "+***-***-***-"
            return cur + res[-4:]
```