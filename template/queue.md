### 225. Implement Stack using Queues

```python
class MyStack:

    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        self.q.append(x)
        x = len(self.q)
        for i in range(x - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not self.q
```

### 346. Moving Average from Data Stream

```python
class MovingAverage:

    def __init__(self, size: int):
        self.q = deque()
        self.n = size
        self.total = 0
        self.sum = 0

    def next(self, val: int) -> float:
        if self.total == self.n:
            x = self.q.popleft()
            self.q.append(val)
            self.sum += val - x
            return self.sum / self.n
        self.q.append(val)
        self.total += 1
        self.sum += val 
        return self.sum / self.total 
```

### 900. RLE Iterator

```python
class RLEIterator:

    def __init__(self, encoding: List[int]):
        self.q = deque()
        for i in range(0, len(encoding), 2):
            if encoding[i] != 0:
                self.q.append((encoding[i], encoding[i + 1]))
        
    def next(self, n: int) -> int:
        res = inf
        while self.q:
            num, val = self.q.popleft()
            if num > n:
                self.q.appendleft((num - n, val))
                res = val
                break 
            elif num == n:
                res = val
                break
            else:
                n -= num 
        return res if res != inf else -1
```