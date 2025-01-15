### 2349. Design a Number Container System

```python
class NumberContainers:

    def __init__(self):
        self.index_to_number = defaultdict(int)
        self.number_to_index = defaultdict(list)

    def change(self, index: int, number: int) -> None:
        self.index_to_number[index] = number 
        heappush(self.number_to_index[number], index)

    def find(self, number: int) -> int:
        h = self.number_to_index[number]
        while h and self.index_to_number[h[0]] != number:
            heappop(h)
        return h[0] if h else -1
```

### 2034. Stock Price Fluctuation 

```python
from sortedcontainers import SortedList

class StockPrice:

    def __init__(self):
        self.d = defaultdict(int)
        self.sl = SortedList()
        self.latest = 0

    def update(self, timestamp: int, price: int) -> None:
        if timestamp in self.d:
            old = self.d[timestamp]
            self.sl.remove(old)
        self.d[timestamp] = price 
        self.sl.add(price)
        self.latest = max(self.latest, timestamp)

    def current(self) -> int:
        return self.d[self.latest]

    def maximum(self) -> int:
        return self.sl[-1]

    def minimum(self) -> int:
        return self.sl[0]
```