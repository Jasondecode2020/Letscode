## Question list

### Math (1)

* [60. Permutation Sequence](#60-permutation-sequence) 1900

### Greedy (1)

* [2086. Minimum Number of Food Buckets to Feed the Hamsters](#2086-minimum-number-of-food-buckets-to-feed-the-hamsters) 1600

### String (1)

* [165. Compare Version Numbers](#165-Compare-Version-Numbers) 1500

### Bit manipulation

* [477. Total Hamming Distance](#477-total-hamming-distance)

### Linked List (1)

* [707. Design Linked List](#707-design-linked-list)

### Graph

* [935. Knight Dialer](#935-knight-dialer)

### 60. Permutation Sequence

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        res, nums = '', list(range(1, n + 1))
        for i in range(1, n + 1):
            index = 0
            cnt = factorial(n - i)
            while k > cnt:
                index += 1
                k -= cnt
            res += str(nums[index])
            nums.pop(index)
        return res
```

### 2086. Minimum Number of Food Buckets to Feed the Hamsters

```python
class Solution:
    def minimumBuckets(self, hamsters: str) -> int:
        n = len(hamsters)
        i = 0
        res = 0
        while i < n:
            if hamsters[i] == 'H':
                if i + 1 < n and hamsters[i + 1] == '.':
                    i += 2
                    res += 1
                elif i - 1 >= 0 and hamsters[i - 1] == '.':
                    res += 1
                else:
                    return -1
            i += 1
        return res
```

### 165. Compare Version Numbers

```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        v1, v2 = version1.split('.'), version2.split('.')
        mx = max(len(v1), len(v2))
        if len(v1) < mx:
            v1.extend(['0'] * (mx - len(v1)))
        if len(v2) < mx:
            v2.extend(['0'] * (mx - len(v2)))
        for a, b in zip(v1, v2):
            if int(a) > int(b):
                return 1
            elif int(a) < int(b):
                return -1
        return 0
```

### 477. Total Hamming Distance

```python
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        d = defaultdict(int)
        n = len(nums)
        for num in nums:
            b = bin(num)[2:].zfill(32)
            for i, c in enumerate(b):
                if c == '1':
                    d[i] += 1
    
        res = 0
        for v in d.values():
            res += v * (n - v)
        return res
```

### 707. Design Linked List

```python
class ListNode:

    def __init__(self, val = 0):
        self.val = val
        self.next = None

class MyLinkedList:

    def __init__(self):
        self.head = ListNode()
        self.size = 0

    def get(self, index: int) -> int:
        if index >= self.size or index < 0:
            return -1
        p = self.head
        for i in range(index + 1):
            p = p.next 
        return p.val 

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size or index < 0:
            return 
        self.size += 1
        p = self.head 
        for i in range(index):
            p = p.next 
        temp = ListNode(val)
        temp.next = p.next 
        p.next = temp 

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return 
        self.size -= 1
        p = self.head 
        for i in range(index):
            p = p.next 
        p.next = p.next.next


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```

### 935. Knight Dialer

```python
class Solution:
    def knightDialer(self, n: int) -> int:
        g = {
            1: [6, 8],
            2: [7, 9],
            3: [4, 8],
            4: [0, 3, 9],
            6: [0, 1, 7],
            7: [2, 6],
            8: [1, 3],
            9: [2, 4],
            0: [4, 6]
        }
        mod = 10 ** 9 + 7
        @cache
        def dfs(i, step):
            if step == n - 1:
                return 1
            res = 0
            for j in g[i]:
                res += dfs(j, step + 1)
            return res
        if n == 1:
            return 10
        res = sum(dfs(i, 0) for i in range(10) if i != 5)
        return res % mod
```