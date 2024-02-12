## template

```python
from collections import Counter
c = Counter(nums) # can compair
a = c.most_common() # [(key, value)]
```

### 169. Majority Element

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        for n in nums:
            if count == 0:
                res = n 
                count += 1
            elif n == res:
                count += 1
            else:
                count -= 1
        return res
```

### 229. Majority Element II

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        A, B, cntA, cntB = None, None, 0, 0
        for n in nums:
            if n == A:
                cntA += 1
            elif n == B:
                cntB += 1
            elif A is None:
                A = n
                cntA += 1
            elif B is None:
                B = n
                cntB += 1
            else:
                cntA -= 1
                cntB -= 1
                if not cntA:
                    A = None
                if not cntB:
                    B = None 
        return [n for n in [A, B] if nums.count(n) > len(nums) // 3]
```

### 2190. Most Frequent Number Following Key In an Array

```python
class Solution:
    def mostFrequent(self, nums: List[int], key: int) -> int:
        c = Counter()
        L = len(nums)
        for i, n in enumerate(nums):
            if i < L - 1 and n == key:
                c[nums[i + 1]] += 1
        return c.most_common()[0][0]
```

### 2248. Intersection of Multiple Arrays

```python
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        n = len(nums)
        c, res = Counter(), []
        for item in nums:
            for i in item:
                c[i] += 1
        for k, v in c.items():
            if v == n:
                res.append(k)
        return sorted(res)
```

### 2068. Check Whether Two Strings are Almost Equivalent

```python
class Solution:
    def checkAlmostEquivalent(self, word1: str, word2: str) -> bool:
        c1, c2 = Counter(word1), Counter(word2)
        for c in ascii_lowercase:
            if abs(c1[c] - c2[c]) > 3:
                return False
        return True
```

### 2287. Rearrange Characters to Make Target String

```python
class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        res = inf
        c1, c2 = Counter(s), Counter(target)
        for c in c2:
            res = min(res, c1[c] // c2[c])
        return res
```

### 2347. Best Poker Hand

```python
class Solution:
    def bestHand(self, ranks: List[int], suits: List[str]) -> str:
        r, s = Counter(ranks), Counter(suits)
        if len(s) == 1:
            return 'Flush'
        for k, v in r.items():
            if v >= 3:
                return 'Three of a Kind'
        for k, v in r.items():
            if v == 2:
                return 'Pair'
        return 'High Card'
```

### 2423. Remove Letter To Equalize Frequency

```python
class Solution:
    def equalFrequency(self, word: str) -> bool:
        for i, c in enumerate(word):
            w = Counter(word[: i] + word[i + 1:])
            if len(set(w.values())) == 1:
                return True
        return False
```

### 447. Number of Boomerangs

```python
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        res = 0
        for x1, y1 in points:
            c = Counter()
            for x2, y2 in points:
                d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
                res += c[d2] * 2
                c[d2] += 1
        return res
```

### 916. Word Subsets

```python
class Solution:
    def wordSubsets(self, words1: List[str], words2: List[str]) -> List[str]:
        c2 = Counter()
        for w2 in words2:
            countWord2 = Counter(w2)
            for c in ascii_lowercase:
                if c in countWord2:
                    c2[c] = max(c2[c], countWord2[c])
        
        res = []
        for w1 in words1:
            c1 = Counter(w1)
            if c1 >= c2:
                res.append(w1)
        return res
```

### 1347. Minimum Number of Steps to Make Two Strings Anagram

```python
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        s, t = Counter(s), Counter(t)
        res = s - t
        return sum(res.values())
```
