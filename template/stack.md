## stack to solve recursive function

### 224. Basic Calculator

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack, res, num, sign = [], 0, 0, 1
        for c in s:
            if c.isdigit():
                num = num * 10 + int(c)
            elif c in ['+', '-']:
                res += sign * num 
                num = 0
                sign = 1 if c == '+' else -1
            elif c == '(':
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ')':
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        res += num * sign 
        return res
```

### 682. Baseball Game

```python
class Solution:
    def calPoints(self, operations: List[str]) -> int:
        s = []
        for o in operations:
            if o[0] not in 'CD+':
                s.append(int(o))
            if o == '+':
                s.append(s[-1] + s[-2])
            if o == 'C':
                s.pop()
            if o == 'D':
                s.append(s[-1] * 2)
        return sum(s)
```

### 232. Implement Queue using Stacks

```python
class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)

    def pop(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self):
        return not self.stack1 and not self.stack2
```