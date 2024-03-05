## template: bit manipulation

* `421. Maximum XOR of Two Numbers in an Array`


### 292. Nim Game

```python
class Solution:
    def canWinNim(self, n: int) -> bool:
        return n % 4 != 0
```

### 294. Flip Game II

```python
class Solution:
    def canWin(self, currentState: str) -> bool:
        @cache
        def dfs(state):
            for i in range(len(currentState) - 1):
                if (state >> i) & 1 or (state >> (i + 1)) & 1:
                    continue
                if not dfs(state | 1 << i | state | 1 << (i + 1)):
                    return True
            return False

        res = 0
        for i in range(len(currentState)):
            if currentState[i] == '-':
                res += 1 << i
        return dfs(res)
```

### 421. Maximum XOR of Two Numbers in an Array

- start from highest bit to lowest bit, use bit mask to check, use two sum to find ans

```python
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        res, mask = 0, 0
        for i in range(32, -1, -1):
            mask |= 1 << i # 10000, 11000, ..., 11111
            ans = res | 1 << i # 10000, 11000, 11100, 11100, 11100
            s = set()
            for n in nums:
                n &= mask
                if n ^ ans in s:
                    res = ans
                    break
                s.add(n)
        return res
```

### 371. Sum of Two Integers

```python
class Solution {
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;
            a ^= b;
            b = carry;
        }
        return a;
    }
}
```

### 191. Number of 1 Bits - (n): n is 32

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        return n.bit_count()
        # return sum(1 for i in range(32) if n & (1 << i))
```

### 191. Number of 1 Bits - O(k): k is the number of 1's

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += 1
            n = n & (n - 1)
        return res
```

### 464. Can I Win

```python
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        @cache
        def dfs(state, total):
            for i in range(1, maxChoosableInteger + 1):
                if (state >> i) & 1:
                    continue
                if total + i >= desiredTotal:
                    return True
                if not dfs(state | (1 << i), total + i):
                    return True
            return False
        if maxChoosableInteger * (maxChoosableInteger + 1) // 2 < desiredTotal:
            return False
        return dfs(0, 0)
```

### 2411. Smallest Subarrays With Maximum Bitwise OR

```python
class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        def ok2remove(count, num):
            for i in range(32):
                if count[i] == 1 and ((num >> i) & 1):
                    return False
            return True

        count = [0] * 32
        n = len(nums)
        res = [0] * n
        j = n - 1
        for i in range(n - 1, -1, -1):
            for k in range(32):
                if (nums[i] >> k) & 1:
                    count[k] += 1
            while j > i and ok2remove(count, nums[j]):
                for k in range(32):
                    if (nums[j] >> k) & 1:
                        count[k] -= 1
                j -= 1
            res[i] = j - i + 1
        return res
```

### 260. Single Number III

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        res = nums[0]
        n = len(nums)
        for i in range(1, n):
            res ^= nums[i]
        lowbit = res & -res
        ans = [0, 0]
        for x in nums:
            if x & lowbit == 0:
                ans[0] ^= x
            else:
                ans[1] ^= x
        return ans
```

### 2917. Find the K-or of an Array

```python
class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        count = [0] * 32
        for n in nums:
            for i in range(32):
                if n & (1 << i):
                    count[i] += 1
        return sum(2 ** i for i in range(32) if count[i] >= k)
```

### 401. Binary Watch

```python
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        res =[]
        for i in range(12): # hour
            for j in range(60): # minute
                if((bin(i)+bin(j)).count('1')==turnedOn): # light number
                   res.append((str(i)+":"+str(j).zfill(2))) # fill zero for minutes  
        return res 
```

### 1318. Minimum Flips to Make a OR b Equal to c

```python
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        res = 0
        for i in range(32):
            z = c & (1 << i)
            x = a & (1 << i)
            y = b & (1 << i)
            if z and x == 0 and y == 0:
                res += 1
            if not z and (x or y):
                if x:
                    res += 1
                if y:
                    res += 1
        return res
```

## xor

### 2683. Neighboring Bitwise XOR

```python
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        res = 0
        for n in derived:
            res ^= n 
        return res == 0
```

### 2429. Minimize XOR

```python
class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        ones = num2.bit_count()
        a = list(bin(num1)[2:].zfill(32))
        res = 0
        for i, c in enumerate(a):
            if c == '1' and ones:
                res += 1 << (32 - i - 1)
                ones -= 1
        if ones == 0:
            return res 
        for i, n in enumerate(a[::-1]):
            if n == '0' and ones:
                res += 1 << i
                ones -= 1 
        return res
```

### 2433. Find The Original Array of Prefix Xor

```python
class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        n, prev = len(pref), 0
        for i in range(1, n):
            prev ^= pref[i - 1]
            pref[i] ^= prev
        return pref
```

### 2396. Strictly Palindromic Number

```python
class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        def check(i, n):
            res = []
            while n:
                res.append(n % i)
                n //= i
            return res

        def reverse(a):
            return a == a[::-1]
        for i in range(2, n - 1):
            a = check(i, n)
            if not reverse(a):
                return False
        return True
```