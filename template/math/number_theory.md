# Template

## 1 Euler

```python
def isPrime(n):
  for i in range(2, int(sqrt(n)) + 1):
      if n % i == 0:
          return False
  return n >= 2
```

## 2 Eratosthenes

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes
primes = ePrime(100)
```

## 3 Prime factors of n

```python
def primeFactors(n):
    res = []
    for i in range(2, int(sqrt(n)) + 1):
        while n % i == 0:
            res.append(i)
            n //= i
    if n >= 2:
        res.append(n)
    return res
```

## 4 Prime factors of array

```python
divisors = defaultdict(list)
def ePrimeFactors(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i, n + 1, i): # starting from i 
                divisors[j].append(i)
                primes[j] = False
ePrimeFactors(1000)
```

## 5 Divisors of n

```python
def divisorsN(self, n):
    res = 0
    for i in range(1, int(sqrt(n)) + 1):
        if n % i == 0:
            if i != n // i:
                res += 2
            else:
                res += 1
    return res
```

## 6 Divisors of array

```python
N = 10 ** 5 + 1
divisors = [[] for i in range(N)]
for i in range(1, N):
    for j in range(i, N, i):
        divisors[j].append(i)
```

## 1 check prime numbers (6)

* [3115. Maximum Prime Difference 1294](#3115-maximum-prime-difference)
* [2614. Prime In Diagonal 1375](#2614-prime-in-diagonal)
* [762. Prime Number of Set Bits in Binary Representation 1383](#762-prime-number-of-set-bits-in-binary-representation)
* [1175. Prime Arrangements 1489](#1175-prime-arrangements)
* [3044. Most Frequent Prime 1737](#3044-most-frequent-prime)
* [866. Prime Palindrome 1938](#866-prime-palindrome)

### 3115. Maximum Prime Difference

```python
class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def isPrime(n):
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return n >= 2

        mn, mx = inf, -inf
        for i, n in enumerate(nums):
            if isPrime(n):
                mn = min(mn, i)
                mx = max(mx, i)
        return mx - mn 
```

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes
primes = ePrime(100)

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        mn, mx = inf, -inf
        for i, n in enumerate(nums):
            if primes[n]:
                mn = min(mn, i)
                mx = max(mx, i)
        return mx - mn 
```

### 2614. Prime In Diagonal

```python
class Solution:
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        def isPrime(n):
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return n >= 2

        n, res = len(nums), 0
        for i in range(n):
            if isPrime(nums[i][i]):
                res = max(res, nums[i][i])
            if isPrime(nums[i][n - i - 1]):
                res = max(res, nums[i][n - i - 1])
        return res
```

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes
primes = ePrime(4 * 10 ** 6)

class Solution:
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        n, res = len(nums), 0
        for i in range(n):
            if primes[nums[i][i]]:
                res = max(res, nums[i][i])
            if primes[nums[i][n - i - 1]]:
                res = max(res, nums[i][n - i - 1])
        return res
```

### 762. Prime Number of Set Bits in Binary Representation

```python
class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def isPrime(n):
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return n >= 2

        res = 0
        for n in range(left, right + 1):
            num = n.bit_count()
            if isPrime(num):
                res += 1
        return res
```

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes
primes = ePrime(32)

class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        res = 0
        for n in range(left, right + 1):
            num = n.bit_count()
            if primes[num]:
                res += 1
        return res
```

### 1175. Prime Arrangements

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes

class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        mod = 10 ** 9 + 7
        primeCount = ePrime(n).count(True)
        return factorial(primeCount) * factorial(n - primeCount) % mod
```

### 3044. Most Frequent Prime

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes
primes = ePrime(10 ** 6)
directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]

class Solution:
    def mostFrequentPrime(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        d = Counter()
        for r in range(R):
            for c in range(C):
                num = mat[r][c]
                if primes[num] and num > 10:
                    d[num] += 1
                for dr, dc in directions:
                    num, row, col  = mat[r][c], r, c
                    for i in range(6):
                        row, col = row + dr, col + dc
                        if 0 <= row < R and 0 <= col < C:
                            num = num * 10 + mat[row][col]
                            if primes[num]:
                                d[num] += 1
        if not d.values():
            return -1
        mx = max(d.values())
        res = sorted([k for k, v in d.items() if v == mx], reverse = True)
        return res[0]            
```

### 866. Prime Palindrome

- All palindrome with even digits is multiple of 11

```python
class Solution:
    def primePalindrome(self, n: int) -> int:
        def isPalindrome(n):
            return str(n) == str(n)[::-1]
        
        def isPrime(n): # include n
            for i in range(2, int(sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return n >= 2
        
        while True:
            if isPalindrome(n) and isPrime(n):
                return n
            n += 1
            if len(str(n)) in [4, 6, 8]:
                n = 10 ** len(str(n)) + 1
```

## 2 preprocess of prime numbers (4)

* [204. Count Primes 1400](#204-count-primes)
* [2761. Prime Pairs With Target Sum 1505](#2761-prime-pairs-with-target-sum)
* [2523. Closest Prime Numbers in Range 1650](#2523-closest-prime-numbers-in-range)
* [2601. Prime Subtraction Operation 1779](#2601-prime-subtraction-operation)

### 204. Count Primes

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes

class Solution:
    def countPrimes(self, n: int) -> int:
        primes = ePrime(n - 1)
        return primes.count(True)
```

### 2761. Prime Pairs With Target Sum

```python
def ePrime(n):
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        primes = ePrime(n)
        res = []
        for i in range(2, n // 2 + 1):
            if primes[i] and primes[n - i]:
                res.append([i, n - i])
        return res
```

### 2523. Closest Prime Numbers in Range

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes

class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        primes = ePrime(right)
        nums = [n for n in range(left, right + 1) if primes[n]]
        res = sorted((nums[i] - nums[i - 1], nums[i - 1], nums[i]) for i in range(1, len(nums)))
        return res[0][1:] if res else [-1, -1]
```

### 2601. Prime Subtraction Operation

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes

class Solution:
    def primeSubOperation(self, nums: List[int]) -> bool:
        nums = [0] + nums
        primes = ePrime(max(nums))
        for i in range(len(nums)):
            for n in range(nums[i] - 1, 1, -1):
                if primes[n] and nums[i] - n > nums[i - 1]:
                    nums[i] -= n
                    break
        return len(nums) == len(set(nums)) and nums == sorted(nums)
```

## 3 prime factors

* [2521. Distinct Prime Factors of Product of Array](#2521-distinct-prime-factors-of-product-of-array)
* [2507. Smallest Value After Replacing With Sum of Prime Factors 1499](#2507-smallest-value-after-replacing-with-sum-of-prime-factors)
* [2584. Split the Array to Make Coprime Products 2159](#2584-split-the-array-to-make-coprime-products)
* [2709. Greatest Common Divisor Traversal 2173](#2709-greatest-common-divisor-traversal)
* [952. Largest Component Size by Common Factor 2272](#952-largest-component-size-by-common-factor)
* [1998. GCD Sort of an Array](#1998-gcd-sort-of-an-array)
* [2862. Maximum Element-Sum of a Complete Subset of Indices](#2862-maximum-element-sum-of-a-complete-subset-of-indices)
* [2818. Apply Operations to Maximize Score]
* [1735. Count Ways to Make Array With Product]
* [2338. Count the Number of Ideal Arrays 2665](#2338-count-the-number-of-ideal-arrays)

### 2521. Distinct Prime Factors of Product of Array

```python
def primeFactors(n):
    res = []
    for i in range(2, int(sqrt(n)) + 1):
        while n % i == 0:
            res.append(i)
            n //= i
    if n >= 2:
        res.append(n)
    return res

class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        s = set()
        for n in nums:
            for f in primeFactors(n):
                s.add(f)
        return len(s)
```

```python
divisors = defaultdict(list)
def ePrimeFactors(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i, n + 1, i): # starting from i 
                divisors[j].append(i)
                primes[j] = False
ePrimeFactors(1000)

class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        s = set()
        for n in nums:
            for f in divisors[n]:
                s.add(f)
        return len(s)
```

### 2507. Smallest Value After Replacing With Sum of Prime Factors

```python
class Solution:
    def smallestValue(self, n: int) -> int:
        def primeFactors(n):
            res = []
            for i in range(2, int(sqrt(n)) + 1):
                while n % i == 0:
                    res.append(i)
                    n //= i 
            if n >= 2:
                res.append(n)
            return sum(res)

        while n != primeFactors(n):
            n = primeFactors(n)
        return n
```

### 2584. Split the Array to Make Coprime Products

```python
class Solution:
    def findValidSplit(self, nums: List[int]) -> int:
        d = defaultdict(int)
        for n in nums:
            for num in divisors[n]:
                d[num] += 1

        pre_d = defaultdict(int)
        for i, n in enumerate(nums):
            if i == len(nums) - 1: return -1
            for num in divisors[n]:
                pre_d[num] += 1
                d[num] -= 1
                if d[num] == 0:
                    d.pop(num)
            if all(v not in d for v in pre_d.keys()):
                return i 
```

### 2709. Greatest Common Divisor Traversal

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2 
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

divisors = defaultdict(list)
def ePrime(n):
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(n + 1):
        if primes[i]:
            for j in range(i, n + 1, i):
                divisors[j].append(i)
                primes[j] = False
ePrime(100001)  

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        uf = UF(max(nums) + 1)
        for n in nums:
            for a, b in pairwise(divisors[n]):
                uf.union(a, b)
        d = Counter()
        if len(nums) == 1:
            return True
        for n in nums:
            if n == 1:
                return False
            res = uf.find(divisors[n][0])
            d[res] += 1
        return max(list(d.values())) == len(nums)
```

### 952. Largest Component Size by Common Factor

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2 
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

divisors = defaultdict(list)
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i, n + 1, i):
                divisors[j].append(i)
                primes[j] = False
ePrime(100001)

class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        uf = UF(max(nums) + 1)
        for i in nums:
            for j, k in pairwise(divisors[i]):
                uf.union(k, j)
        d = defaultdict(int)
        for i in nums:
            if i == 1:
                continue
            res = uf.find(divisors[i][0])
            d[res] += 1
        return max(d.values())
```


### 1998. GCD Sort of an Array

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2 
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

divisors = defaultdict(list)
def ePrime(n):
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(n + 1):
        if primes[i]:
            for j in range(i, n + 1, i):
                divisors[j].append(i)
                primes[j] = False
ePrime(100001)  
       
class Solution:
    def gcdSort(self, nums: List[int]) -> bool:
        uf = UF(max(nums) + 1)
        for n in nums:
            for a, b in pairwise(divisors[n]):
                uf.union(a, b)

        d = defaultdict(list)
        for i, n in enumerate(nums):
            res = uf.find(divisors[n][0])
            d[res].append((n, i))
        res = [0] * len(nums)
        for v in d.values():
            for (a, _), i in zip(sorted(v), sorted(i for _, i in v)):
                res[i] = a
        return res == sorted(nums)
```

### 2862. Maximum Element-Sum of a Complete Subset of Indices

```python
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        squares = [i * i for i in range(1, 101)]
        nums = [0] + nums 
        n = len(nums)
        
        res = 0
        for i in range(1, n):
            ans = 0
            flag = 0
            for j in squares:
                if i * j < n:
                    flag += 1
                    ans += nums[i * j]
            if flag >= 2:
                res = max(res, ans)
        return max(res, max(nums))
```

### 2338. Count the Number of Ideal Arrays

```python
MOD, MX = 10 ** 9 + 7, 10 ** 4 + 1
primeFreq = [[] for _ in range(MX)] 
for j in range(2, MX):
    n = j
    res = []
    for i in range(2, int(n ** 0.5) + 1):
        while n % i == 0:
            res.append(i)
            n //= i
    if n >= 2:
        res.append(n)
    primeFreq[j].extend(Counter(res).values())

class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        res = 0
        for i in range(1, maxValue + 1):
            mul = 1
            for k in primeFreq[i]:
                mul = mul * comb(n + k - 1, k) % MOD
            res += mul
        return res % MOD
```

## 4 factorial

### 172. Factorial Trailing Zeroes

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        res = 0
        for i in range(5, n + 1, 5):
            while i % 5 == 0:
                i //= 5
                res += 1
        return res 
```

## 5 divisors

* [2427. Number of Common Factors](#2427-number-of-common-factors)
* [1952. Three Divisors](#1952-three-divisors)
* [1492. The kth Factor of n](#1492-the-kth-factor-of-n)
* [507. Perfect Number](#507-perfect-number)
* [1390. Four Divisors](#1390-four-divisors)
* [1362. Closest Divisors](#1362-closest-divisors)
* [829. Consecutive Numbers Sum](#829-consecutive-numbers-sum)
* [952. Largest Component Size by Common Factor](#952-largest-component-size-by-common-factor)
* [1627. Graph Connectivity With Threshold](#1627-graph-connectivity-with-threshold)
* [2183. Count Array Pairs Divisible by K](#)

### 2427. Number of Common Factors

```python
class Solution:
    def commonFactors(self, a: int, b: int) -> int:
        mn = gcd(a, b)
        res = 0
        for i in range(1, int(sqrt(mn)) + 1):
            if mn % i == 0:
                if i != mn // i:
                    res += 2
                else:
                    res += 1
        return res
```

### 1952. Three Divisors

```python
class Solution:
    def isThree(self, n: int) -> bool:
        res = 0
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0:
                if i != n // i:
                    res += 2
                else:
                    res += 1
        return res == 3
```

### 1492. The kth Factor of n

```python
class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        def divisorsN(n):
            res = []
            for i in range(1, int(sqrt(n)) + 1):
                if n % i == 0:
                    if i != n // i:
                        res.extend([i, n // i])
                    else:
                        res.append(i)
            return sorted(res)
        res = divisorsN(n)
        return res[k - 1] if k <= len(res) else -1
```

### 507. Perfect Number

```python
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        res = 0
        for i in range(1, int(sqrt(num)) + 1):
            if num % i == 0:
                if i != num // i:
                    res += i + num // i
                else:
                    res += i
        return res - num == num
```

### 1390. Four Divisors

```python
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        def check(n):
            res = set()
            for i in range(1, int(sqrt(n)) + 1):
                if n % i == 0:
                    res.add(i)
                    res.add(n // i)
                    if len(res) > 4:
                        break
            if len(res) == 4:
                return sum(list(res))
            return 0

        res = 0
        for n in nums:
            res += check(n)
        return res
```

### 1362. Closest Divisors

```python
class Solution:
    def closestDivisors(self, num: int) -> List[int]:
        res = [-inf, inf]
        for n in range(num + 1, num + 3):
            for i in range(1, int(sqrt(n)) + 1):
                if n % i == 0:
                    a, b = i, n // i
                    if b - a < res[1] - res[0]:
                        res = [a, b]
        return res
```

### 829. Consecutive Numbers Sum

```python
class Solution:
    def consecutiveNumbersSum(self, n: int) -> int:
        res, n = 0, n * 2
        k = 1
        while k * k < n:
            if n % k == 0 and (n // k - (k - 1)) % 2 == 0:
                res += 1
            k += 1
        return res
```

### 952. Largest Component Size by Common Factor

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2 
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

divisors = defaultdict(list)
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i, n + 1, i):
                divisors[j].append(i)
                primes[j] = False
ePrime(100001)

class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        uf = UF(max(nums) + 1)
        for i in nums:
            for j, k in pairwise(divisors[i]):
                uf.union(k, j)
        d = defaultdict(int)
        for i in nums:
            if i == 1:
                continue
            res = uf.find(divisors[i][0])
            d[res] += 1
        return max(d.values())
```

### 1627. Graph Connectivity With Threshold

```python
class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n 

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2 
            self.rank[p2] += self.rank[p1]

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        uf = UF(n + 1)
        res = []
        for i in range(threshold + 1, n + 1, 1): 
            for j in range(i, n + 1, i):
                uf.union(i, j)
        for x, y in queries:                       
            res.append(uf.isConnected(x, y))
        return res
```

### 2183. Count Array Pairs Divisible by K

```python
N = 10 ** 5 + 1
divisors = [[] for i in range(N)]
for i in range(1, N):
    for j in range(i, N, i):
        divisors[j].append(i)

class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        # [6,3,3,4,5], k = 6, gcd(4, 6) = 2, k = 3
        # a * b = k
        res = 0
        c = Counter()
        for n in nums:
            res += c[k // gcd(n, k)]
            for factor in divisors[n]:
                c[factor] += 1
        return res
```


## 6 gcd

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

### 914. X of a Kind in a Deck of Cards

```python
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        v = Counter(deck).values()
        x = reduce(gcd, v)
        return x > 1
```

### 1071. Greatest Common Divisor of Strings

```python
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        for i in range(min(len(str1), len(str2)), 0, -1):
            if (len(str1) % i) == 0 and (len(str2) % i) == 0:
                if str1[: i] * (len(str1) // i) == str1 and str1[: i] * (len(str2) // i) == str2:
                    return str1[: i]
        return ''
```

### 2001. Number of Pairs of Interchangeable Rectangles

```python
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        ratio = Counter()
        res = 0
        for a, b in rectangles: # 4, 8 => 4 // 4, 8 // 4 => (1, 2)
            ratio[a // gcd(a, b), b // gcd(a, b)] += 1
        for n in ratio.values():
            res += n * (n - 1) // 2
        return res
```

## 7 lcm

* [2413. Smallest Even Multiple](#2413-smallest-even-multiple)
* [2470. Number of Subarrays With LCM Equal to K](#2470-number-of-subarrays-with-lcm-equal-to-k)
* [2447. Number of Subarrays With GCD Equal to K](#2447-number-of-subarrays-with-gcd-equal-to-k)
* [2197. Replace Non-Coprime Numbers in Array](#2197-replace-non-coprime-numbers-in-array)

### 2413. Smallest Even Multiple

```python
class Solution:
    def smallestEvenMultiple(self, n: int) -> int:
        return n if n % 2 == 0 else n * 2
```

### 2470. Number of Subarrays With LCM Equal to K

```python
class Solution:
    def subarrayLCM(self, nums: List[int], k: int) -> int:
        # 1
        # [3,6,2,7,1]
        res, n = 0, len(nums)
        for i in range(n):
            ans = 1
            for j in range(i, n):
                ans = lcm(ans, nums[j])
                if ans == k:
                    res += 1
                if k % ans:
                    break
        return res
```

### 2447. Number of Subarrays With GCD Equal to K

```python
class Solution:
    def subarrayGCD(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(len(nums)):
            g = 0
            for j in range(i, len(nums)):
                g = gcd(g, nums[j])
                if g % k: break
                if g == k: ans += 1
        return ans
```

### 2197. Replace Non-Coprime Numbers in Array

```python
class Solution:
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        stack = []
        for n in nums:
            stack.append(n)
            while len(stack) > 1:
                n1, n2 = stack[-1], stack[-2]
                g = gcd(n1, n2)
                if g == 1:
                    break
                x = stack.pop()
                stack[-1] = lcm(x, stack[-1])
        return stack
```

## 8 prime with each other

* [2748. Number of Beautiful Pairs](#2748-number-of-beautiful-pairs)

### 2748. Number of Beautiful Pairs

```python
class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(int(str(nums[i])[0]), int(str(nums[j])[-1])) == 1:
                    res += 1
        return res
```

### 1447. Simplified Fractions

```python
class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        res = []
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                if gcd(i, j) == 1:
                    res.append(str(i) + '/' + str(j))
        return res
```

## 9 mod

## 10 others