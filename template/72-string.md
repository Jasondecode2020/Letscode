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