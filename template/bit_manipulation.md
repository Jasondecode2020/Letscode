## bit manipulation

1. n.bit_length(): the length of the binary number
2. n.bit_count(): the number of set bits
3. num1 |= num1 + 1: set lowbit of 0 to 1
4. num1 &= num1 - 1: set lowbit of 1 to 0
5. x & (-x): lowbit 


### 1 basic

* [3370. Smallest Number With All Set Bits](#3370-smallest-number-with-all-set-bits)
* [461. Hamming Distance](#461-hamming-distance)
* [2220. Minimum Bit Flips to Convert Number](#2220-minimum-bit-flips-to-convert-number)
* [1356. Sort Integers by The Number of 1 Bits](#1356-sort-integers-by-the-number-of-1-bits)
* [3226. Number of Bit Changes to Make Two Integers Equal](#3226-number-of-bit-changes-to-make-two-integers-equal)
* [1342. Number of Steps to Reduce a Number to Zero](#1342-number-of-steps-to-reduce-a-number-to-zero)
* [476. Number Complement](#476-number-complement)
* [1009. Complement of Base 10 Integer](#1009-complement-of-base-10-integer)
* [868. Binary Gap](#868-binary-gap)
* [3211. Generate Binary Strings Without Adjacent Zeros](#3211-generate-binary-strings-without-adjacent-zeros)
* [2917. Find the K-or of an Array](#2917-find-the-k-or-of-an-array)
* [693. Binary Number with Alternating Bits](#693-binary-number-with-alternating-bits)
* [2657. Find the Prefix Common Array of Two Arrays](#2657-find-the-prefix-common-array-of-two-arrays)
* [231. Power of Two](#231-power-of-two)
* [342. Power of Four](#342-power-of-four)
* [191. Number of 1 Bits](#191-number-of-1-bits)
* [2595. Number of Even and Odd Bits](#2595-number-of-even-and-odd-bits)
* [338. Counting Bits](#338-counting-bits)
* [2997. Minimum Number of Operations to Make Array XOR Equal to K](#2997-minimum-number-of-operations-to-make-array-xor-equal-to-k)

### 2 xor

* [1486. XOR Operation in an Array](#1486-xor-operation-in-an-array)
* [1720. Decode XORed Array](#1720-decode-xored-array)
* [2433. Find The Original Array of Prefix Xor](#2433-find-the-original-array-of-prefix-xor)
* [1310. XOR Queries of a Subarray](#1310-xor-queries-of-a-subarray)
* [2683. Neighboring Bitwise XOR](#2683-neighboring-bitwise-xor)
* [1829. Maximum XOR for Each Query](#1829-maximum-xor-for-each-query)
* [1442. Count Triplets That Can Form Two Arrays of Equal XOR](#1442-count-triplets-that-can-form-two-arrays-of-equal-xor)
* [2527. Find Xor-Beauty of Array](#2527-find-xor-beauty-of-array)
* [2317. Maximum XOR After Operations](#2317-maximum-xor-after-operations)
* [2433. Find The Original Array of Prefix Xor](#2433-find-the-original-array-of-prefix-xor)
* [2588. Count the Number of Beautiful Subarrays]()

### 3 or/and


* [421. Maximum XOR of Two Numbers in an Array](#421-maximum-xor-of-two-numbers-in-an-array)
* [2275. Largest Combination With Bitwise AND Greater Than Zero](#2275-largest-combination-with-bitwise-and-greater-than-zero)


### 3370. Smallest Number With All Set Bits

<details markdown=1><summary markdown='span'>Answer</summary>
```python
class Solution:
    def smallestNumber(self, n: int) -> int:
        return (1 << n.bit_length()) - 1
```
</details>


### 461. Hamming Distance

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return (x ^ y).bit_count()
```

### 2220. Minimum Bit Flips to Convert Number

```python
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        return (start ^ goal).bit_count()
```

### 1356. Sort Integers by The Number of 1 Bits

```python
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        a = [(n.bit_count(), n) for n in arr]
        a.sort()
        return [x[1] for x in a]
```

### 3226. Number of Bit Changes to Make Two Integers Equal

```python
class Solution:
    def minChanges(self, n: int, k: int) -> int:
        return -1 if n & k != k else (n ^ k).bit_count()
```

### 1342. Number of Steps to Reduce a Number to Zero

```python
class Solution:
    def numberOfSteps(self, num: int) -> int:
        res = 0
        while num:
            if num % 2 == 0:
                num //= 2
            else:
                num -= 1
            res += 1
        return res 
```

### 476. Number Complement

```python
class Solution:
    def findComplement(self, num: int) -> int:
        mask = (1 << num.bit_length()) - 1
        return num ^ mask 
```

### 1009. Complement of Base 10 Integer

```python
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        if n == 0: return 1
        mask = (1 << n.bit_length()) - 1
        return n ^ mask 
```

### 868. Binary Gap

```python
class Solution:
    def binaryGap(self, n: int) -> int:
        s = bin(n)[2:]
        a = [i for i, c in enumerate(s) if c == '1']
        if len(a) < 2:
            return 0
        return max(b - a for a, b in pairwise(a))
```

### 3211. Generate Binary Strings Without Adjacent Zeros

```python
class Solution:
    def validStrings(self, n: int) -> List[str]:
        def valid(s):
            a = [i for i, c in enumerate(s) if c == '0']
            if len(a) > 1 and min(b - a for a, b in pairwise(a)) == 1:
                return False
            return True

        res = []
        for i in range(2 ** n):
            s = bin(i)[2:].zfill(n)
            if valid(s):
                res.append(s)
        return res 
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
        return sum(1 << i for i in range(32) if count[i] >= k)
```

### 693. Binary Number with Alternating Bits

```python
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        L = n.bit_length()
        if n % 2 == 0:
            return n == int('10' * (L // 2), 2)
        else:
            return n == int('1' + '01' * ((L - 1) // 2), 2)
```

### 2657. Find the Prefix Common Array of Two Arrays

```python
class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        n = len(A)
        res = []
        for i in range(1, n + 1):
            s = set(A[:i]) & set(B[:i])
            if s:
                res.append(len(s))
            else:
                res.append(0)
        return res
```

### 231. Power of Two

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n & (n - 1) == 0 if n != 0 else False
```

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n.bit_count() == 1 if n > 0 else False
```

### 342. Power of Four

```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        if n <= 0:
            return False
        while n != 1:
            if n % 4 == 0:
                n //= 4 
            else:
                return False 
        return True
```

### 191. Number of 1 Bits

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        return n.bit_count()
```

### 2595. Number of Even and Odd Bits

```python
class Solution:
    def evenOddBit(self, n: int) -> List[int]:
        s = bin(n)[2:]
        even = odd = 0
        for i, c in enumerate(list(s)[::-1]):
            if c == '1':
                if i % 2 == 0:
                    even += 1
                else:
                    odd += 1
        return [even, odd]
```

### 338. Counting Bits

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = []
        for i in range(n + 1):
            res.append(i.bit_count())
        return res 

class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i // 2] + 1 if i % 2 else dp[i // 2]
        return dp
```

### 1486. XOR Operation in an Array

```python
class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        nums = [start + 2 * i for i in range(n)]
        res = 0
        for n in nums:
            res ^= n 
        return res 
```

### 1720. Decode XORed Array

```python
class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        res = [first]
        for n in encoded:
            res.append(res[-1] ^ n)
        return res 
```

### 2433. Find The Original Array of Prefix Xor

```python
class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        for i in range(len(pref) - 1, 0, -1):
            pref[i] ^= pref[i - 1]
        return pref
```

### 1310. XOR Queries of a Subarray

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        arr = [0] + arr
        for i in range(1, len(arr)):
            arr[i] ^= arr[i - 1]
        res = []
        for s, e in queries:
            res.append(arr[e + 1] ^ arr[s])
        return res
```

### 2683. Neighboring Bitwise XOR

```python
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        return reduce(xor, derived) == 0
```

### 1829. Maximum XOR for Each Query

```python
class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        n = len(nums)
        mask = (1 << maximumBit) - 1
        for i in range(1, n):
            nums[i] ^= nums[i - 1]
        res = [0] * n 
        for i in range(n):
            res[i] = nums[i] ^ mask
        return res[::-1]
```

### 2997. Minimum Number of Operations to Make Array XOR Equal to K

```python
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        ans = reduce(xor, nums)
        return (ans ^ k).bit_count()
```

### 1442. Count Triplets That Can Form Two Arrays of Equal XOR

```python
class Solution:
    def countTriplets(self, arr: List[int]) -> int:
        pre = [0] + arr 
        n = len(pre)
        for i in range(1, n):
            pre[i] ^= pre[i - 1]
        
        res = 0
        for i in range(1, n):
            for j in range(i + 1, n):
                if pre[i - 1] ^ pre[j] == 0:
                    res += j - i
        return res 
```

### 2527. Find Xor-Beauty of Array

```python
class Solution:
    def xorBeauty(self, nums: List[int]) -> int:
        # [a, b]
        # (a | a) & a = a
        # (a | a) & b = a & b
        # (a | b) & a = 0
        # (b | a) & a = 
        # (b | b) & b = b
        # (b | b) & a = b & a
        # (b | a) & b = 0
        # (a | b) & b = 
        return reduce(xor, nums)
```

### 2317. Maximum XOR After Operations 

```python
class Solution:
    def maximumXOR(self, nums: List[int]) -> int:
        return reduce(or_, nums)
```

### 2588. Count the Number of Beautiful Subarrays

```python
class Solution:
    def beautifulSubarrays(self, nums: List[int]) -> int:
        d = defaultdict(int)
        d[0] = 1
        total = 0
        res = 0
        for n in nums:
            total ^= n
            res += d[total]
            d[total] += 1
        return res 
```

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

### 2275. Largest Combination With Bitwise AND Greater Than Zero

```python
class Solution:
    def largestCombination(self, candidates: List[int]) -> int:
        n = len(candidates)
        res = [0] * 32
        for c in candidates:
            s = bin(c)[2:][::-1]
            for i, num in enumerate(s):
                if num == '1':
                    res[i] += 1
        return max(res)
```