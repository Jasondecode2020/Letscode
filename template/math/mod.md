## mod

### 2156. Find Substring With Given Hash Value

```python
class Solution:
    def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
        mult = pow(power, k - 1, modulo)   # power^(k-1) mod modulo
        n = len(s)
        pos = -1   # first position
        h = 0   # hash val
        for i in range(n - 1, n - k - 1, -1):
            h = (h * power + (ord(s[i]) - ord('a') + 1)) % modulo
        if h == hashValue:
            pos = n - k
        # from backward
        for i in range(n - k - 1, -1, -1):
            h = ((h - (ord(s[i+k]) - ord('a') + 1) * mult % modulo) * power) % modulo # python no need to plus modulo
            h = (h + (ord(s[i]) - ord('a') + 1)) % modulo
            if h == hashValue:
                pos = i
        return s[pos:pos+k]
```

### 507. Perfect Number

```python
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num == 1:
            return False
        res = 0
        for i in range(1, int(sqrt(num)) + 1):
            if num % i == 0:
                res += i + num // i
        return res - num == num
```

### 1523. Count Odd Numbers in an Interval Range

```python
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        diff = high - low 
        if diff % 2 == 0:
            if low % 2 and high % 2:
                return diff // 2 + 1
            else:
                return diff // 2
        else:
            return diff // 2 + 1
```