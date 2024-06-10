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