## sorting: to sort arr simulation ans

- 768. Max Chunks To Make Sorted II
- 769. Max Chunks To Make Sorted


### 768. Max Chunks To Make Sorted II

```python
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        c, res = defaultdict(int), 0
        for a, b in zip(arr, sorted(arr)):
            c[a] += 1
            if c[a] == 0:
                del c[a]
            c[b] -= 1
            if c[b] == 0:
                del c[b]
            if len(c) == 0:
                res += 1
        return res
```

### 769. Max Chunks To Make Sorted

```python
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        c, res = defaultdict(int), 0
        for a, b in zip(arr, sorted(arr)):
            c[a] += 1
            if c[a] == 0:
                del c[a]
            c[b] -= 1
            if c[b] == 0:
                del c[b]
            if len(c) == 0:
                res += 1
        return res
```

### 912. Sort an Array

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # [5,2,3,1]
        if len(nums) <= 1:
            return nums
        
        index = random.randint(0, len(nums) - 1)
        pivot = nums[index]

        greater = [n for n in nums if n > pivot]
        equal = [n for n in nums if n == pivot]
        less = [n for n in nums if n < pivot]
        return self.sortArray(less) + equal + self.sortArray(greater)
```

## bucket sort

### 347. Top K Frequent Elements

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # method 1: dict
        # c = Counter(nums)
        # print(c.most_common(2))
        # return [item[0] for item in c.most_common(k)]

        # method 2: heap
        # c = Counter(nums)
        # pq = [(v, k) for k, v in c.items()]
        # return [item[1] for item in nlargest(k, pq)]

        # method 3: bucket sort
        c, n, res = Counter(nums), len(nums), []
        bucket = [[] for _ in range(n + 1)]
        for x, v in c.items():
            bucket[v].append(x)

        for v in range(n, -1, -1):
            res.extend(bucket[v])
        return res[: k]
```

### 912. Sort an Array

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        # [5,2,3,1]
        if len(nums) <= 1:
            return nums
        
        index = random.randint(0, len(nums) - 1)
        pivot = nums[index]

        greater = [n for n in nums if n > pivot]
        equal = [n for n in nums if n == pivot]
        less = [n for n in nums if n < pivot]
        return self.sortArray(less) + equal + self.sortArray(greater)
```