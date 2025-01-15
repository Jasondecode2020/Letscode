## 31 is the same as 556

* [31. Next Permutation](#31-next-permutation)
* [556. Next Greater Element III](#556-next-greater-element-iii)

### 31. Next Permutation

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        pivot = None 
        n = len(nums)
        for i in range(n - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                pivot = i - 1
                break 
        if pivot == None:
            nums.reverse()
            return 
        
        for i in range(n - 1, 0, -1):
            if nums[i] > nums[pivot]:
                nums[i], nums[pivot] = nums[pivot], nums[i]
                break
        
        nums[pivot + 1:] = nums[pivot + 1:][::-1]
```

### 556. Next Greater Element III

```python
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        # 1234321
        nums = list(str(n))
        n = len(nums)
        pivot = None 
        for i in range(n - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                pivot = i - 1
                break 
        if pivot == None:
            return -1

        for i in range(n - 1, 0, -1):
            if nums[i] > nums[pivot]:
                nums[pivot], nums[i] = nums[i], nums[pivot]
                break 

        nums[pivot + 1:] = nums[pivot + 1:][::-1]
        res = int(''.join(nums))
        return res if res <= 2 ** 31 - 1 else -1
```

### 284. Peeking Iterator

```python
class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.res = None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if self.res:
            return self.res 
        self.res = self.iterator.next()
        return self.res
        

    def next(self):
        """
        :rtype: int
        """
        if self.res:
            cur = self.res 
            self.res = None
            return cur
        return self.iterator.next()
        

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.res:
            return True
        return self.iterator.hasNext()
```

### 1139. Largest 1-Bordered Square

```python
class Solution:
    def largest1BorderedSquare(self, grid: List[List[int]]) -> int:
        def hasBorder(r, c, n):
            top = grid[r][c: c + n]
            bottom = grid[r + n - 1][c: c + n]
            left = [grid[i][c] for i in range(r, r + n)]
            right = [grid[i][c + n - 1] for i in range(r, r + n)]
            if sum(top + bottom + left + right) == 4 * n:
                return True
            return False 

        def check(n):
            for r in range(R - n + 1):
                for c in range(C - n + 1):
                    if hasBorder(r, c, n):
                        return True
            return False

        R, C = len(grid), len(grid[0])
        res = 0
        N = min(len(grid), len(grid[0]))
        for i in range(N, 0, -1):
            if check(i):
                return i * i 
        return 0
```

### 1044. Longest Duplicate Substring

```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        def check(m):
            visited = set()
            for i in range(0, len(s) - m + 1):
                word = s[i: i + m]
                if word in visited:
                    if len(self.ans) < len(word):
                        self.ans = word 
                    return True
                visited.add(word)
            return False 
            
        self.ans = ''
        l, r, res = 0, len(s), 0
        while l <= r:
            m = (l + r) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return self.ans
```

### 1062. Longest Repeating Substring

```python
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        def check(m):
            visited = set()
            for i in range(0, len(s) - m + 1):
                word = s[i: i + m]
                if word in visited:
                    return True
                visited.add(word)
            return False 
            
        l, r, res = 0, len(s), 0
        while l <= r:
            m = (l + r) // 2
            if check(m):
                res = m 
                l = m + 1
            else:
                r = m - 1
        return res
```

### 519. Random Flip Matrix

```python
class Solution:

    def __init__(self, m: int, n: int):
        self.R = m - 1
        self.C = n - 1
        self.visited = set()

    def flip(self) -> List[int]:
        while True:
            x, y = randint(0, self.R), randint(0, self.C)
            if (x, y) not in self.visited:
                self.visited.add((x, y))
                return [x, y]

    def reset(self) -> None:
        self.visited.clear()
```

### 738. Monotone Increasing Digits

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        ones, res = 111111111, 0
        for i in range(9):
            while res + ones > n:
                ones //= 10
            res += ones 
        return res 
```

### 1236. Web Crawler

```python
class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        url = startUrl.split('/')
        url[1] = '//'
        include_pattern = ''.join(url[:3])
        q = deque([startUrl])
        visited = set([startUrl])
        while q:
            start_url = q.popleft()
            urls = htmlParser.getUrls(start_url)
            for url in urls:
                if url not in visited:
                    if url.startswith(include_pattern):
                        visited.add(url)
                        q.append(url)
        return list(visited)
```