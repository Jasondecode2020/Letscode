## stack 

### basics

* [1441. Build an Array With Stack Operations](#1441-build-an-array-with-stack-operations) 1180
* [844. Backspace String Compare](#844-Backspace-String-Compare) 1228
* [682. Baseball Game](#682-baseball-game) 1250
* [2390. Removing Stars From a String](#2390-removing-stars-from-a-string) 1348
* [1472. Design Browser History](#1472-design-browser-history) 1454
* [946. Validate Stack Sequences](#946-validate-stack-sequences) 1462
* [71. Simplify Path](#71-simplify-path) 1500

### 1441. Build an Array With Stack Operations

```python
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        res = []
        i = 1
        j = 0
        while i <= n and j < len(target):
            if i == target[j]:
                res.append('Push')
                i += 1
                j += 1
            else:
                res.extend(['Push', 'Pop'])
                i += 1
        return res
```

```java
class Solution {
    public List<String> buildArray(int[] target, int n) {
        int i = 0;
        int j = 1;
        List<String> res = new ArrayList<String>();
        while (i < target.length && j < n + 1) {
            if (target[i] == j) {
                res.add("Push");
                i += 1;
                j += 1;
            } else {
                res.add("Push");
                res.add("Pop");
                j += 1;
            }
                
        }
        return res;
    }
}
```

### 844. Backspace String Compare

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def getString(s):
            stack = []
            for c in s:
                if c != '#':
                    stack.append(c)
                elif stack:
                    stack.pop()
            return stack 
        return getString(s) == getString(t)
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

### 1472. Design Browser History

```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.stack = [homepage]
        self.queue = deque()
    def visit(self, url: str) -> None:
        self.stack.append(url)
        self.queue = deque()

    def back(self, steps: int) -> str:
        while len(self.stack) > 1 and steps:
            url = self.stack.pop()
            self.queue.appendleft(url)
            steps -= 1
        return self.stack[-1]

    def forward(self, steps: int) -> str:
        while self.queue and steps:
            url = self.queue.popleft()
            self.stack.append(url)
            steps -= 1
        return self.stack[-1]
```

### 946. Validate Stack Sequences

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack, i = [], 0
        for n in pushed:
            stack.append(n)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1
        return not stack
```

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

```java
class Solution {
    public String simplifyPath(String path) {
        String[] paths = path.split("/");
        Stack<String> stack = new Stack<String>();
        for (String s: paths) {
            if ("..".equals(s)) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                if (s.length() > 0 && !".".equals(s)) {
                    stack.push(s);
                }
            }
        }
        String res = "";
        for (String s: stack) res = res + "/" + s;
        return res.isEmpty() ? "/" : res;
    }
}
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

### 1190. Reverse Substrings Between Each Pair of Parentheses

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        stack = ['']
        for c in s:
            if c == '(':
                stack.append('')
            elif c == ')':
                last = stack.pop()
                stack[-1] += last[::-1]
            else:
                stack[-1] += c 
        return stack[0]
```


### 1003. Check If Word Is Valid After Substitutions

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            stack.append(c)
            if len(stack) > 2 and ''.join(stack[-3:]) == 'abc':
                for i in range(3):
                    stack.pop()
        return not stack
```

### 1249. Minimum Remove to Make Valid Parentheses

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        for i, c in enumerate(s):
            if not c.islower():
                if stack and stack[-1][0] == '(' and c == ')':
                    stack.pop()
                else:
                    stack.append((c, i))
        
        seen = set([i for c, i in stack])
        res = ''
        for i, c in enumerate(s):
            if i not in seen:
                res += c
        return res
```