### 2047. Number of Valid Words in a Sentence

```python
class Solution:
    def countValidWords(self, sentence: str) -> int:
        res = 0
        def check(word):
            if re.match("[a-z]*([a-z]-[a-z])?[a-z]*[,.!]?$", word):
                return True
            return False

        for w in sentence.split():
            if check(w):
                res += 1
        return res
```

### 65. Valid Number

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        reg = '^[+-]?((\d+\.?)|(\d*\.\d+))([eE][+-]?\d+)?$'
        if re.match(reg, s):
            return True
        return False
```