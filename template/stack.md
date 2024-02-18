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

### 71. Simplify Path

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        path = path.split('/')

        for p in path:
            if p == '..':
                if stack:
                    stack.pop()
            elif p and p != '.':
                stack.append(p)
        return '/' + '/'.join(stack)
```

### 1006. Clumsy Factorial

```python
class Solution:
    def clumsy(self, n: int) -> int:
        op = 0
        stack = [n]
        for i in range(n - 1, 0, -1):
            if op == 0:
                stack.append(stack.pop() * i)
            elif op == 1:
                stack.append(int(stack.pop() / i))
            elif op == 2:
                stack.append(i)
            elif op == 3:
                stack.append(-i)
            op = (op + 1) % 4
        return sum(stack)
```

### 227. Basic Calculator II

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack, pre, num = [], '+', 0
        for i, c in enumerate(s):
            if c.isdigit():
                num = num * 10 + int(c)
            if i == len(s) - 1 or c in '+-*/':
                if pre == '+':
                    stack.append(num)
                elif pre == '-':
                    stack.append(-num)
                elif pre == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                pre = c
                num = 0
        return sum(stack)
```

### 1556. Thousand Separator

```python
class Solution:
    def thousandSeparator(self, n: int) -> str:
        stack = list(str(n))
        res = ''
        while len(stack) >= 3:
            n1, n2, n3 = stack.pop(), stack.pop(), stack.pop()
            res = '.' + str(n3) + str(n2) + str(n1) + res
        return ''.join(stack) + res if stack else res[1:]
```

### 2390. Removing Stars From a String

```python
class Solution:
    def removeStars(self, s: str) -> str:
        stack = []
        for c in s:
            if c != '*':
                stack.append(c)
            else:
                stack and stack.pop()
        return ''.join(stack)
```

### 921. Minimum Add to Make Parentheses Valid

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        stack = []
        for p in s:
            if stack and stack[-1] == '(' and p == ')':
                stack.pop()
            else:
                stack.append(p)
        return len(stack)
```