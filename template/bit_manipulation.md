## bit manipulation

<details markdown=1>
<summary markdown='span'>Answer</summary>
<li>1. n.bit_length(): the length of the binary number</li>
<li>2. n.bit_count(): the number of set bits</li>
<li>3. num1 |= num1 + 1: set lowbit of 0 to 1</li>
<li>4. num1 &= num1 - 1: set lowbit of 1 to 0</li>
<li>5. x & (-x): lowbit</li>
<li>6. 01-Trie</li>
<pre>
class TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children = [None, None]

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, x):
        cur = self.root
        for i in range(30, -1, -1):
            n = x >> i & 1
            if cur.children[n] is None:
                cur.children[n] = TrieNode()
            cur = cur.children[n]

    def search(self, x):
        cur = self.root
        res = 0
        for i in range(30, -1, -1):
            n = x >> i & 1
            if cur.children[n ^ 1]:
                res |= 1 << i
                cur = cur.children[n ^ 1]
            else:
                cur = cur.children[n]
        return res
</pre>
</details>


### 1 basic (18)

* [3370. Smallest Number With All Set Bits](#3370-smallest-number-with-all-set-bits)
* [3226. Number of Bit Changes to Make Two Integers Equal](#3226-number-of-bit-changes-to-make-two-integers-equal)
* [1356. Sort Integers by The Number of 1 Bits](#1356-sort-integers-by-the-number-of-1-bits)
* [461. Hamming Distance](#461-hamming-distance)
* [2220. Minimum Bit Flips to Convert Number](#2220-minimum-bit-flips-to-convert-number)
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

### 2 xor (15)

* [1486. XOR Operation in an Array](#1486-xor-operation-in-an-array)
* [1720. Decode XORed Array](#1720-decode-xored-array)
* [2433. Find The Original Array of Prefix Xor](#2433-find-the-original-array-of-prefix-xor)
* [1310. XOR Queries of a Subarray](#1310-xor-queries-of-a-subarray)
* [2683. Neighboring Bitwise XOR](#2683-neighboring-bitwise-xor)
* [1829. Maximum XOR for Each Query](#1829-maximum-xor-for-each-query)
* [2997. Minimum Number of Operations to Make Array XOR Equal to K](#2997-minimum-number-of-operations-to-make-array-xor-equal-to-k)
* [1442. Count Triplets That Can Form Two Arrays of Equal XOR](#1442-count-triplets-that-can-form-two-arrays-of-equal-xor)
* [2429. Minimize XOR](#2429-minimize-xor)
* [2527. Find Xor-Beauty of Array](#2527-find-xor-beauty-of-array)
* [2317. Maximum XOR After Operations](#2317-maximum-xor-after-operations)
* [2588. Count the Number of Beautiful Subarrays](#2588-count-the-number-of-beautiful-subarrays)
* [2564. Substring XOR Queries](#2564-substring-xor-queries)
* [1734. Decode XORed Permutation](#1734-decode-xored-permutation)
* [2857. Count Pairs of Points With Distance k](#2857-count-pairs-of-points-with-distance-k)
* [1803. Count Pairs With XOR in a Range](#1803-count-pairs-with-xor-in-a-range)
* [3215. Count Triplets with Even XOR Set Bits II](#3215-count-triplets-with-even-xor-set-bits-ii)

### 3 or/and (9)

* [2980. Check if Bitwise OR Has Trailing Zeros](#2980-check-if-bitwise-or-has-trailing-zeros)
* [1318. Minimum Flips to Make a OR b Equal to c](#1318-minimum-flips-to-make-a-or-b-equal-to-c)
* [2419. Longest Subarray With Maximum Bitwise AND](#2419-longest-subarray-with-maximum-bitwise-and)
* [2871. Split Array Into Maximum Number of Subarrays](#2871-split-array-into-maximum-number-of-subarrays)
* [2401. Longest Nice Subarray](#2401-longest-nice-subarray)
* [2680. Maximum OR](#2680-maximum-or)
* [3133. Minimum Array End](#3133-minimum-array-end)
* [3108. Minimum Cost Walk in Weighted Graph](#3108-minimum-cost-walk-in-weighted-graph)
* [3125. Maximum Number That Makes Result of Bitwise AND Zero](#3215-count-triplets-with-even-xor-set-bits-ii)
* [3117. Minimum Sum of Values by Dividing Array](#3117-minimum-sum-of-values-by-dividing-array)

### 4 LogTrick (and/or) (7)

* [3097. Shortest Subarray With OR at Least K II](#3097-shortest-subarray-with-or-at-least-k-ii)
* [2411. Smallest Subarrays With Maximum Bitwise OR](#2411-smallest-subarrays-with-maximum-bitwise-or)
* [3209. Number of Subarrays With AND Value of K](#3209-number-of-subarrays-with-and-value-of-k)
* [3171. Find Subarray With Bitwise OR Closest to K](#3171-find-subarray-with-bitwise-or-closest-to-k)
* [1521. Find a Value of a Mysterious Function Closest to Target](#1521-find-a-value-of-a-mysterious-function-closest-to-target)
* [898. Bitwise ORs of Subarrays](#898-bitwise-ors-of-subarrays)
* [2654. Minimum Number of Operations to Make All Array Elements Equal to 1](#2654-minimum-number-of-operations-to-make-all-array-elements-equal-to-1)

### 5 split bit/check together

* [477. Total Hamming Distance](#477-total-hamming-distance)
* [1863. Sum of All Subset XOR Totals](#1863-sum-of-all-subset-xor-totals)
* [2425. Bitwise XOR of All Pairings](#2425-bitwise-xor-of-all-pairings)
* [2275. Largest Combination With Bitwise AND Greater Than Zero](#2275-largest-combination-with-bitwise-and-greater-than-zero)
* [3153. Sum of Digit Differences of All Pairs](#3153-sum-of-digit-differences-of-all-pairs)
* [1835. Find XOR Sum of All Pairs Bitwise AND](#1835-find-xor-sum-of-all-pairs-bitwise-and)
* [2505. Bitwise OR of All Subsequence Sums](#2505-bitwise-or-of-all-subsequence-sums)

### 6 try and put

* [421. Maximum XOR of Two Numbers in an Array](#421-maximum-xor-of-two-numbers-in-an-array)
* [2935. Maximum Strong Pair XOR II](#2935-maximum-strong-pair-xor-ii)

### 7 equation

* [1835. Find XOR Sum of All Pairs Bitwise AND](#1835-find-xor-sum-of-all-pairs-bitwise-and)
* [2354. Number of Excellent Pairs](#2354-number-of-excellent-pairs)

### 8 skill

* [2546. Apply Bitwise Operations to Make Strings Equal](#2546-apply-bitwise-operations-to-make-strings-equal)
* [1558. Minimum Numbers of Function Calls to Make Target Array](#1558-minimum-numbers-of-function-calls-to-make-target-array)
* [2571. Minimum Operations to Reduce an Integer to 0](#2571-minimum-operations-to-reduce-an-integer-to-0)
* [3315. Construct the Minimum Bitwise Array II](#3315-construct-the-minimum-bitwise-array-ii)
* [2568. Minimum Impossible OR](#2568-minimum-impossible-or)
* [2509. Cycle Length Queries in a Tree](#2509-cycle-length-queries-in-a-tree)

### 9 others

* [136. Single Number](#136-single-number)
* [2275. Largest Combination With Bitwise AND Greater Than Zero](#2275-largest-combination-with-bitwise-and-greater-than-zero)


### 3370. Smallest Number With All Set Bits

```python
class Solution:
    def smallestNumber(self, n: int) -> int:
        return (1 << n.bit_length()) - 1
```

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

### 2564. Substring XOR Queries

```python
class Solution:
    def substringXorQueries(self, s: str, queries: List[List[int]]) -> List[List[int]]:
        d = defaultdict(list)
        n = len(s)
        for i in range(n):
            for j in range(i + 1, min(n + 1, i + 31)):
                d[int(s[i: j], 2)].append((j - i, i, j - 1))
        for k, v in d.items():
            d[k] = sorted(v)[0]
        
        res = []
        for a, b in queries:
            x = a ^ b 
            if x in d:
                res.append(d[x][1:])
            else:
                res.append([-1, -1])
        return res
```

### 1734. Decode XORed Permutation

```python
class Solution:
    def decode(self, encoded: List[int]) -> List[int]:
        n = len(encoded)
        # ABCDE
        x = 0
        for i in range(1, n + 2):
            x ^= i 
        # BCDE
        y = 0
        for i in range(1, n, 2):
            y ^= encoded[i]
        
        first = x ^ y
        res = [first]
        for n in encoded:
            x = n ^ res[-1]
            res.append(x)
        return res 
```

### 2857. Count Pairs of Points With Distance k

```python
class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # x1 ^ x2 + y1 ^ y2 = k
        # 0 <= x1 ^ x2 <= k
        # 0 <= y1 ^ y2 <= k 
        # x1 ^ x2 = i => y1 ^ y2 = k - i 
        # x1 = x2 ^ i, y1 = y2 ^ (k - i)
        res = 0
        d = Counter()
        for x, y in coordinates:
            for i in range(k + 1):
                res += d[(x ^ i, y ^ (k - i))]
            d[(x, y)] += 1
        return res 
```

### 1803. Count Pairs With XOR in a Range

```python
class TrieNode:
    __slot__ = ('children', 'cnt')
    
    def __init__(self):
        self.children = [None, None]
        self.cnt = 0

class Trie:
    
    def __init__(self):
        self.root = TrieNode()

    def insert(self, x):
        cur = self.root 
        for i in range(15, -1, -1):
            n = (x >> i) & 1
            if cur.children[n] is None:
                cur.children[n] = TrieNode()
            cur = cur.children[n]
            cur.cnt += 1

    def search(self, x, limit):
        cur = self.root 
        res = 0
        for i in range(15, -1, -1):
            if not cur:
                break 
            n = (x >> i) & 1
            if (limit >> i) & 1:
                if cur.children[n]:
                    res += cur.children[n].cnt 
                cur = cur.children[n ^ 1]
            else:
                cur = cur.children[n]
        return res 

class Solution:
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        trie = Trie()
        res = 0
        for n in nums:
            res += trie.search(n, high + 1) - trie.search(n, low)
            trie.insert(n)
        return res 
```

### 3215. Count Triplets with Even XOR Set Bits II

```python
class Solution:
    def tripletCount(self, a: List[int], b: List[int], c: List[int]) -> int:
        c1, c2, c3 = Counter(), Counter(), Counter()
        def check(x, c):
            for n in x:
                x = n.bit_count()
                if x % 2 == 0:
                    c[2] += 1
                else:
                    c[1] += 1

        for x, c in [(a, c1), (b, c2), (c, c3)]:
            check(x, c)

        res = c1[2] * c2[2] * c3[2]
        res += c1[1] * c2[1] * c3[2] + c1[1] * c2[2] * c3[1] + c1[2] * c2[1] * c3[1]
        return res
```

### 3117. Minimum Sum of Values by Dividing Array

```python
class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        @cache
        def dfs(i, j, and_):
            if i == n:
                return 0 if j == m else inf 
            if j == m:
                return 0 if i == n else inf 
            and_ &= nums[i]
            res = dfs(i + 1, j, and_)
            if and_ == andValues[j]:
                res = min(res, dfs(i + 1, j + 1, -1) + nums[i])
            return res 
        res = dfs(0, 0, -1)
        return res if res != inf else -1
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

### 2980. Check if Bitwise OR Has Trailing Zeros

```python
class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        return sum(1 for n in nums if n % 2 == 0) > 1
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

### 2419. Longest Subarray With Maximum Bitwise AND

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        res = 0
        mx = max(nums)
        for a, b in groupby(nums):
            if a == mx:
                res = max(res, len(list(b)))
        return res 
```

### 2871. Split Array Into Maximum Number of Subarrays

```python
class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # nums = [1,0,2,0,1,2]
        mask = (1 << 32) - 1
        ans, res = mask, 0
        for i, n in enumerate(nums):
            ans &= n
            if ans == 0:
                res += 1
                ans = mask
        return res if res != 0 else 1
```

### 2401. Longest Nice Subarray

```python
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        count = [0] * 32
        l, res = 0, 0
        for r, n in enumerate(nums):
            for i in range(32):
                if n & (1 << i):
                    count[i] += 1
            while any(x > 1 for x in count):
                for i in range(32):
                    if nums[l] & (1 << i):
                        count[i] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res
```

### 2680. Maximum OR

```python
class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pre, suf = nums[::], nums[::]
        for i in range(1, n):
            pre[i] |= pre[i - 1]
        for i in range(n - 2, -1, -1):
            suf[i] |= suf[i + 1]
        pre, suf = [0] + pre + [0], [0] + suf + [0]
        res = 0
        for i in range(n):
            for j in range(k + 1):
                res = max(res, pre[i] | suf[i + 2] | (nums[i] << k))
        return res
```

### 3133. Minimum Array End

```python
class Solution:
    def minEnd(self, n: int, x: int) -> int:
        l = bin(x)[2:].zfill(64)
        a = list(l)
        L = len(l)
        s = bin(n - 1)[2:]
        i, j = L - 1, len(s) - 1
        while i >= 0 and j >= 0:
            if a[i] == '0' and s[j] == '0':
                i -= 1
                j -= 1
            elif a[i] == '0' and s[j] == '1':
                a[i] = '1'
                i -= 1
                j -= 1
            else:
                i -= 1
        return int(''.join(a), 2)
```

### 3108. Minimum Cost Walk in Weighted Graph

```python
class UF:

    def __init__(self, n):
        self.parent = list(range(n))
        self.and_ = [-1] * n 

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2, w):
        p1, p2 = self.find(n1), self.find(n2)
        self.and_[p1] &= w & self.and_[p2]
        self.parent[p2] = p1

class Solution:
    def minimumCost(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        uf = UF(n)
        for u, v, w in edges:
            uf.union(u, v, w)
        res = [-1] * len(query)
        for i, (a, b) in enumerate(query):
            if uf.find(a) == uf.find(b):
                res[i] = uf.and_[uf.find(a)]
        return res 
```

### 1521. Find a Value of a Mysterious Function Closest to Target

```python
class Solution:
    def closestToTarget(self, arr: List[int], target: int) -> int:
        ans = inf
        for i, x in enumerate(arr):
            ans = min(ans, abs(x - target))  # 单个元素也算子数组
            j = i - 1
            while j >= 0 and arr[j] & x != arr[j]:
                arr[j] &= x
                ans = min(ans, abs(arr[j] - target))
                j -= 1
        return ans
```

### 3171. Find Subarray With Bitwise OR Closest to K

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans = inf
        for i, x in enumerate(nums):
            ans = min(ans, abs(x - k))  # 单个元素也算子数组
            j = i - 1
            while j >= 0 and nums[j] | x != nums[j]:
                nums[j] |= x
                ans = min(ans, abs(nums[j] - k))
                j -= 1
        return ans
```

### 3209. Number of Subarrays With AND Value of K

```python
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        res = 0
        for i, x in enumerate(nums):
            j = i - 1
            while j >= 0 and nums[j] & x != nums[j]:
                nums[j] &= x 
                j -= 1
            res += bisect_right(nums, k, 0, i + 1) - bisect_left(nums, k, 0, i + 1)
        return res
```

### 3097. Shortest Subarray With OR at Least K II

```python
class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        res = inf 
        for i, x in enumerate(nums):
            if x >= k:
                return 1
            j = i - 1
            while j >= 0 and nums[j] | x != nums[j]:
                nums[j] |= x
                if nums[j] >= k:
                    res = min(res, i - j + 1)
                j -= 1
        return res if res != inf else -1
```

### 898. Bitwise ORs of Subarrays

```python
class Solution:
    def subarrayBitwiseORs(self, arr: List[int]) -> int:
        res = set()
        for i, x in enumerate(arr):
            res.add(x)
            j = i - 1
            while j >= 0 and arr[j] | x != arr[j]:
                arr[j] |= x
                res.add(arr[j])
                j -= 1
        return len(res)
```

### 2411. Smallest Subarrays With Maximum Bitwise OR

```python
class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        res = [0] * len(nums)
        for i, x in enumerate(nums):
            res[i] = 1
            j = i - 1
            while j >= 0 and nums[j] | x != nums[j]:
                nums[j] |= x 
                res[j] = i - j + 1
                j -= 1
        return res 
```

### 2654. Minimum Number of Operations to Make All Array Elements Equal to 1

```python
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        if reduce(gcd, nums) != 1:
            return -1

        n = len(nums)
        if nums.count(1) > 0:
            return sum(1 for n in nums if n != 1)

        mn_size = inf 
        for i in range(n):
            for j in range(i + 1, n):
                if reduce(gcd, nums[i: j + 1]) == 1:
                    mn_size = min(mn_size, j - i + 1)
        return mn_size + n - 2 
```

### 477. Total Hamming Distance

```python
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(30):
            v = sum((val >> i) & 1 for val in nums)
            res += v * (n - v)
        return res
```

### 1863. Sum of All Subset XOR Totals

```python
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        self.res, n = 0, len(nums)
        def backtrack(idx, val):
            if idx == n:
                self.res += val
                return
            backtrack(idx + 1, val)
            backtrack(idx + 1, val ^ nums[idx])
        backtrack(0, 0)
        return self.res
```

### 2425. Bitwise XOR of All Pairings

```python
class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        nums = nums1 * (len(nums2) % 2) + nums2 * (len(nums1) % 2)
        res = 0
        for n in nums:
            res ^= n 
        return res 
```

### 2275. Largest Combination With Bitwise AND Greater Than Zero

```python
class Solution:
    def largestCombination(self, candidates: List[int]) -> int:
        res = 0
        for i in range(30):
            c = sum((c >> i) & 1 for c in candidates)
            res = max(res, c)
        return res
```

### 3153. Sum of Digit Differences of All Pairs

```python
class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # nums = [13,23,12]
        n = len(str(nums[0]))
        dp = [[0] * 10 for i in range(n)]
        nums = [str(n) for n in nums]
        for num in nums:
            for i, c in enumerate(num[::-1]):
                dp[i][int(c)] += 1
        res = 0 
        for arr in dp:
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    res += arr[i] * arr[j]
        return res
```

### 1835. Find XOR Sum of All Pairs Bitwise AND

```python
class Solution:
    def getXORSum(self, arr1: List[int], arr2: List[int]) -> int:
        res = 0
        for k in range(30, -1, -1):
            c1 = sum(1 for n in arr1 if n & (1 << k))
            c2 = sum(1 for n in arr2 if n & (1 << k))
            if c1 % 2 and c2 % 2:
                res |= (1 << k)
        return res
```

### 2505. Bitwise OR of All Subsequence Sums

```python
class Solution:
    def subsequenceSumOr(self, nums: List[int]) -> int:
        pre, res = 0, 0
        for n in nums:
            pre += n 
            res |= n | pre 
        return res 
```

### 421. Maximum XOR of Two Numbers in an Array

```python
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        res = mask = 0
        for i in range(32, -1, -1):
            mask |= (1 << i)
            ans = res | (1 << i)
            s = set()
            for n in nums:
                n &= mask 
                if n ^ ans in s:
                    res = ans 
                    break
                s.add(n)
        return res 
```

```python
class TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children = [None, None]

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, x):
        cur = self.root
        for i in range(30, -1, -1):
            n = x >> i & 1
            if cur.children[n] is None:
                cur.children[n] = TrieNode()
            cur = cur.children[n]

    def search(self, x):
        cur = self.root
        res = 0
        for i in range(30, -1, -1):
            n = x >> i & 1
            if cur.children[n ^ 1]:
                res |= 1 << i
                cur = cur.children[n ^ 1]
            else:
                cur = cur.children[n]
        return res

class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        trie = Trie()
        for n in nums:
            trie.insert(n)
        return max(trie.search(n) for n in nums)
```

### 2935. Maximum Strong Pair XOR II

```python
class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        nums.sort()
        res = mask = 0
        for i in range(20, -1, -1):
            mask |= (1 << i)
            ans = res | (1 << i)
            d = {}
            for n in nums:
                new_n = n & mask 
                if new_n ^ ans in d and 2 * d[new_n ^ ans] >= n:
                    res = ans 
                    break
                d[new_n] = n 
        return res 
```

### 2354. Number of Excellent Pairs

```python
class Solution:
    def countExcellentPairs(self, nums: List[int], k: int) -> int:
        d = Counter([n.bit_count() for n in set(nums)])
        res = 0
        for k1, v1 in d.items():
            for k2, v2 in d.items():
                if k1 + k2 >= k:
                    res += v1 * v2 
        return res
```

### 2546. Apply Bitwise Operations to Make Strings Equal

```python
class Solution:
    def makeStringsEqual(self, s: str, target: str) -> bool:
        return ('1' in s) == ('1' in target)
```

### 1558. Minimum Numbers of Function Calls to Make Target Array

```python
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        res = 0
        while sum(nums):
            if all(n % 2 == 0 for n in nums):
                res += 1
                nums = [n // 2 for n in nums]
            else:
                res += sum(1 for n in nums if n % 2)
                nums = [n - 1 if n % 2 else n for n in nums]
        return res
```

### 2571. Minimum Operations to Reduce an Integer to 0

```python
class Solution:
    def minOperations(self, n: int) -> int:
        @cache
        def dfs(x):
            if x & (x - 1) == 0:
                return 1
            lowbit = x & -x 
            return 1 + min(dfs(x + lowbit), dfs(x - lowbit))
        return dfs(n)
```

### 3315. Construct the Minimum Bitwise Array II

```python
class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        for i, n in enumerate(nums):
            if n == 2:
                nums[i] = -1
            else:
                t = ~n
                lowbit = t & (-t)
                nums[i] ^= lowbit >> 1
        return nums
```

### 2568. Minimum Impossible OR

```python
class Solution:
    def minImpossibleOR(self, nums: List[int]) -> int:
        s = set(nums)
        for i in range(32):
            x = 1 << i
            if x not in s:
                return x
```

### 2509. Cycle Length Queries in a Tree

```python
class Solution:
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        for i, (a, b) in enumerate(queries):
            res = 1
            while a != b:
                if a > b:
                    a //= 2
                else:
                    b //= 2
                res += 1
            queries[i] = res
        return queries 
```

### 136. Single Number

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(xor, nums)
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