```python
def baseConvertor(n, k):
    res = 0
    while n:
        d, m = divmod(n, k)
        res = 10 * res + m
        n = d 
    return res 
```