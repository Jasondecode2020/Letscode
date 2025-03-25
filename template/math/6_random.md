### 382. Linked List Random Node

```python
class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.nums = []
        while head:
            self.nums.append(head.val)
            head = head.next 

    def getRandom(self) -> int:
        return random.choice(self.nums)
```

### 398. Random Pick Index

```python
class Solution:

    def __init__(self, nums: List[int]):
        self.d = defaultdict(list)
        for i, v in enumerate(nums):
            self.d[v].append(i)

    def pick(self, target: int) -> int:
        return choice(self.d[target])
```

```python
class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        index = self.d[target]
        n = len(index)
        res = 0
        for i, idx in enumerate(index):
            if randrange(i + 1) == 0:
                res = idx
        return res
```

### 497. Random Point in Non-overlapping Rectangles

```python
class Solution:

    def __init__(self, rects: List[List[int]]):
        self.rects = rects
        self.w = [(y2-y1+1) * (x2-x1+1) for x1, y1, x2, y2 in self.rects]

    def pick(self) -> List[int]:
        n = len(self.rects)
        i = random.choices(range(n), self.w)[0]
        rect = self.rects[i]
        y = randint(rect[1], rect[-1])
        x = randint(rect[0], rect[2])
        return [x, y]
```