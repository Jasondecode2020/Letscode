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

* [2521. Distinct Prime Factors of Product of Array]()

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

## 4 factorial

## 5 divisors

## 6 gcd

## 7 lcm

## 8 prime with each other

## 9 mod

## 10 others