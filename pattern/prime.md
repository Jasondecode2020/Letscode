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