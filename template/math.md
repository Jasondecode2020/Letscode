### 1780. Check if Number is a Sum of Powers of Three

```python
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        s = set()
        while n:
            sign = False
            for i in range(15, -1, -1):
                if n - 3 ** i >= 0 and i not in s:
                    n -= 3 ** i 
                    s.add(i)
                    sign = True
                    break
            if not sign:
                break
        return n == 0
```

### 1276. Number of Burgers with No Waste of Ingredients

```python
class Solution:
    def numOfBurgers(self, tomatoSlices: int, cheeseSlices: int) -> List[int]:
        z1, z2 = tomatoSlices, cheeseSlices
        if z1 - 2 * z2 >= 0 and (z1 - 2 * z2) % 2 == 0 and 4 * z2 - z1 >= 0 and (4 * z2 - z1) % 2 == 0:
            return [(z1 - 2 * z2) // 2, (4 * z2 - z1) // 2]
        return []
```