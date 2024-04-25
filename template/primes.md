## template 1: prime factors

```python
def primeFactors(n):
    res = []
    for i in range(2, int(n ** 0.5) + 1):
        while n % i == 0:
            res.append(i)
            n //= i
    if n >= 2:
        res.append(n)
    return sum(res)
```

## template: divisors

```python
divisors = defaultdict(list)
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i, n + 1, i):
                divisors[j].append(i)
                primes[j] = False
```

### 2521. Distinct Prime Factors of Product of Array

```python
class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        s = set()
        for i in nums:
            for j in range(2, int(i ** 0.5) + 1):
                while i % j == 0:
                    s.add(j)
                    i //= j
            if i >= 2:
                s.add(i)
        return len(s)
```

### 2507. Smallest Value After Replacing With Sum of Prime Factors

- same as: 2521

```python
class Solution:
    def smallestValue(self, n: int) -> int:
        def primeFactors(n):
            res = []
            for i in range(2, int(n ** 0.5) + 1):
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

## template 2: prime numbers

```python
def ePrime(n): # include n
    primes = [False] * 2 + [True] * (n - 1)
    for i in range(2, n + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False
    return primes
```

### 204. Count Primes

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        def ePrime(n):
            primes = [False] * 2 + [True] * (n - 1)
            for i in range(2, n + 1):
                if primes[i]:
                    for j in range(i * i, n + 1, i):
                        primes[j] = False
            return primes
        return ePrime(n - 1).count(True)
```

### 1175. Prime Arrangements

```python
class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        def ePrime(n): # include n
            primes = [False] * 2 + [True] * (n - 1)
            for i in range(2, n + 1):
                if primes[i]:
                    for j in range(i * i, n + 1, i):
                        primes[j] = False
            return primes

        mod = 10 ** 9 + 7
        primeCount = ePrime(n).count(True)
        return factorial(primeCount) * factorial(n - primeCount) % mod
```

### 2601. Prime Subtraction Operation

```python
class Solution:
    def primeSubOperation(self, nums: List[int]) -> bool:
        def ePrime(n): # include n
            primes = [False] * 2 + [True] * (n - 1)
            for i in range(2, n + 1):
                if primes[i]:
                    for j in range(i * i, n + 1, i):
                        primes[j] = False
            return primes

        nums = [0] + nums
        primes = ePrime(max(nums))
        for i in range(len(nums)):
            for j in range(nums[i] - 1, 1, -1):
                if primes[j] and nums[i] - j > nums[i - 1]:
                    nums[i] -= j
                    break
        return len(nums) == len(set(nums)) and nums == sorted(nums)
```

### 2761. Prime Pairs With Target Sum

```python
class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def ePrime(n): # include n
            primes = [False] * 2 + [True] * (n - 1)
            for i in range(2, n + 1):
                if primes[i]:
                    for j in range(i * i, n + 1, i):
                        primes[j] = False
            return primes
        
        primes = ePrime(n)
        res = []
        for i in range(2, n + 1):
            j = n - i
            if primes[i] and primes[j] and i <= j:
                res.append([i, j])
        return res
```

### 762. Prime Number of Set Bits in Binary Representation

```python
class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def ePrime(n): # include n
            primes = [False] * 2 + [True] * (n - 1)
            for i in range(2, n + 1):
                if primes[i]:
                    for j in range(i * i, n + 1, i):
                        primes[j] = False
            return primes
        
        primes, res = ePrime(32), 0
        for n in range(left, right + 1):
            c = n.bit_count()
            if primes[c]:
                res += 1
        return res
```

### 2523. Closest Prime Numbers in Range

```python
class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        def ePrime(n): # include n
            primes = [False] * 2 + [True] * (n - 1)
            for i in range(2, n + 1):
                if primes[i]:
                    for j in range(i * i, n + 1, i):
                        primes[j] = False
            return primes
        
        primes, res = ePrime(right), [-1, -1]
        nums = [n for n in range(left, right + 1) if primes[n]]
        ans = []
        for i in range(1, len(nums)):
            ans.append([nums[i] - nums[i - 1], nums[i - 1], nums[i]])
        ans.sort()
        if ans:
            res = ans[0][1:]
        return res
```

## template 3: isPrime

```python
def isPrime(n): # include n
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            return False
    return n >= 2
```

### 2614. Prime In Diagonal

```python
class Solution:
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        def isPrime(n): # include n
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