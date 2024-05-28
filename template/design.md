## design

* [705. Design HashSet](#705. Design HashSet)
* [560. Subarray Sum Equals K](#560-Subarray-Sum-Equals-K)


### 705. Design HashSet

```python
class MyHashSet:

    def __init__(self):
        self.buckets = 1009
        self.table = [[] for _ in range(self.buckets)]

    def hash(self, key):
        return key % self.buckets
    
    def add(self, key):
        hashkey = self.hash(key)
        if key in self.table[hashkey]:
            return
        self.table[hashkey].append(key)
        
    def remove(self, key):
        hashkey = self.hash(key)
        if key not in self.table[hashkey]:
            return
        self.table[hashkey].remove(key)

    def contains(self, key):
        hashkey = self.hash(key)
        return key in self.table[hashkey]
```

### 2043. Simple Bank System

```python
class Bank:

    def __init__(self, balance: List[int]):
        self.balance = balance

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if account1 - 1 < len(self.balance) and self.balance[account1 - 1] >= money:
            if account2 - 1 < len(self.balance):
                self.balance[account1 - 1] -= money
                self.balance[account2 - 1] += money
                return True
        return False
    def deposit(self, account: int, money: int) -> bool:
        if account - 1 < len(self.balance):
            self.balance[account - 1] += money
            return True
        return False

    def withdraw(self, account: int, money: int) -> bool:
        if account - 1 < len(self.balance) and self.balance[account - 1] >= money:
            self.balance[account - 1] -= money
            return True
        return False


# Your Bank object will be instantiated and called as such:
# obj = Bank(balance)
# param_1 = obj.transfer(account1,account2,money)
# param_2 = obj.deposit(account,money)
# param_3 = obj.withdraw(account,money)
```

### 1429. First Unique Number

```python
class FirstUnique:

    def __init__(self, nums: List[int]):
        self.q = deque([])
        self.d = Counter(nums)
        for n in nums:
            if self.d[n] == 1:
                self.q.append(n)

    def showFirstUnique(self) -> int:
        while self.q:
            n = self.q[0]
            if self.d[n] == 1:
                return n 
            else:
                self.q.popleft()
        return -1

    def add(self, value: int) -> None:
        if value not in self.d:
            self.d[value] = 1
            self.q.append(value)
        else:
            self.d[value] += 1


# Your FirstUnique object will be instantiated and called as such:
# obj = FirstUnique(nums)
# param_1 = obj.showFirstUnique()
# obj.add(value)
```

### 1166. Design File System

```python
class FileSystem:

    def __init__(self):
        self.d = defaultdict(int)

    def createPath(self, path: str, value: int) -> bool:
        if path in self.d:
            return False
        n = len(path)
        for i in range(n - 1, -1, -1):
            if path[i] == '/':
                j = i 
                break 
        prefix = path[:j]
        if not prefix or prefix in self.d:
            self.d[path] = value
            return True
        return False

    def get(self, path: str) -> int:
        return self.d[path] if path in self.d else -1



# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.createPath(path,value)
# param_2 = obj.get(path)
```

### 1472. Design Browser History

```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.stack = [homepage]
        self.forword_history = deque()

    def visit(self, url: str) -> None:
        self.stack.append(url)
        self.forword_history = deque()

    def back(self, steps: int) -> str:
        while steps > 0 and len(self.stack) > 1:
            cur = self.stack.pop()
            self.forword_history.appendleft(cur)
            steps -= 1
        return self.stack[-1]

    def forward(self, steps: int) -> str:
        while steps > 0 and self.forword_history:
            url = self.forword_history.popleft()
            self.stack.append(url)
            steps -= 1
        return self.stack[-1]

# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```