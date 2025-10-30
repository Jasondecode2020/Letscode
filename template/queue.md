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
### 346. Moving Average from Data Stream

```python
class MovingAverage:

    def __init__(self, size: int):
        self.q = deque()
        self.size = size

    def next(self, val: int) -> float:
        self.q.append(val)
        if len(self.q) > self.size:
            self.q.popleft()
        return sum(self.q) / len(self.q)
```

### 933. Number of Recent Calls

```python
class RecentCounter:

    def __init__(self):
        self.q = deque()

    def ping(self, t: int) -> int:
        self.q.append(t)
        while self.q and t - self.q[0] > 3000:
            self.q.popleft()
        return len(self.q)
```

### 362. Design Hit Counter

Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).

Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive roughly at the same time.

Implement the HitCounter class:

HitCounter() Initializes the object of the hit counter system.
void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., the past 300 seconds).
 

Example 1:

Input
["HitCounter", "hit", "hit", "hit", "getHits", "hit", "getHits", "getHits"]
[[], [1], [2], [3], [4], [300], [300], [301]]
Output
[null, null, null, null, 3, null, 4, 3]

Explanation
HitCounter hitCounter = new HitCounter();
hitCounter.hit(1);       // hit at timestamp 1.
hitCounter.hit(2);       // hit at timestamp 2.
hitCounter.hit(3);       // hit at timestamp 3.
hitCounter.getHits(4);   // get hits at timestamp 4, return 3.
hitCounter.hit(300);     // hit at timestamp 300.
hitCounter.getHits(300); // get hits at timestamp 300, return 4.
hitCounter.getHits(301); // get hits at timestamp 301, return 3.
 

Constraints:

1 <= timestamp <= 2 * 109
All the calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing).
At most 300 calls will be made to hit and getHits.
 

Follow up: What if the number of hits per second could be huge? Does your design scale?

```python
class HitCounter:

    def __init__(self):
        self.hits = deque()

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        while self.hits and (timestamp - self.hits[0]) >= 300:
            self.hits.popleft()
        return len(self.hits)
```

### 379. Design Phone Directory

Design a phone directory that initially has maxNumbers empty slots that can store numbers. The directory should store numbers, check if a certain slot is empty or not, and empty a given slot.

Implement the PhoneDirectory class:

PhoneDirectory(int maxNumbers) Initializes the phone directory with the number of available slots maxNumbers.
int get() Provides a number that is not assigned to anyone. Returns -1 if no number is available.
bool check(int number) Returns true if the slot number is available and false otherwise.
void release(int number) Recycles or releases the slot number.
 

Example 1:

Input
["PhoneDirectory", "get", "get", "check", "get", "check", "release", "check"]
[[3], [], [], [2], [], [2], [2], [2]]
Output
[null, 0, 1, true, 2, false, null, true]

Explanation
PhoneDirectory phoneDirectory = new PhoneDirectory(3);
phoneDirectory.get();      // It can return any available phone number. Here we assume it returns 0.
phoneDirectory.get();      // Assume it returns 1.
phoneDirectory.check(2);   // The number 2 is available, so return true.
phoneDirectory.get();      // It returns 2, the only number that is left.
phoneDirectory.check(2);   // The number 2 is no longer available, so return false.
phoneDirectory.release(2); // Release number 2 back to the pool.
phoneDirectory.check(2);   // Number 2 is available again, return true.
 

Constraints:

1 <= maxNumbers <= 104
0 <= number < maxNumbers
At most 2 * 104 calls will be made to get, check, and release.

```python
class PhoneDirectory:

    def __init__(self, maxNumbers: int):
        self.s = set(range(maxNumbers))

    def get(self) -> int:
        return self.s.pop() if self.s else -1

    def check(self, number: int) -> bool:
        return number in self.s

    def release(self, number: int) -> None:
        self.s.add(number)
```

### 1429. First Unique Number

```python
class FirstUnique:

    def __init__(self, nums: List[int]):
        self.d = Counter(nums)
        self.q = deque(nums)
    
    def showFirstUnique(self) -> int:
        while self.q:
            if self.d[self.q[0]] == 1:
                return self.q[0]
            else:
                self.q.popleft()
        return -1

    def add(self, value: int) -> None:
        self.d[value] += 1
        self.q.append(value)
```