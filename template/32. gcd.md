## gcd

```python
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
```

### 914. X of a Kind in a Deck of Cards

```python
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        v = Counter(deck).values()
        x = reduce(gcd, v)
        return x > 1
```

### 1979. Find Greatest Common Divisor of Array

```python
class Solution:
    def findGCD(self, nums: List[int]) -> int:
        return gcd(max(nums), min(nums))
```

### 2807. Insert Greatest Common Divisors in Linked List

```python
class Solution:
    def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur.next:
            cur.next = ListNode(gcd(cur.val, cur.next.val), cur.next)
            cur = cur.next.next
        return head
```

### 1250. Check If It Is a Good Array

```python
class Solution:
    def isGoodArray(self, nums: List[int]) -> bool:
        return reduce(gcd, nums) == 1
```

### 2344. Minimum Deletions to Make Array Divisible

```python
class Solution:
    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        nums.sort()
        g = reduce(gcd, numsDivide)
        for i, v in enumerate(nums):
            if g % v == 0:
                return i
        return -1
```