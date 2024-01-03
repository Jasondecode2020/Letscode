## template 1: find minimum value
```python
def fn(arr):
    def check(x):
        # some code
        return BOOLEAN

    l = MINIMUM_POSSIBLE_ANSWER
    r = MAXIMUM_POSSIBLE_ANSWER
    while l <= r:
        m = l + (r - l) // 2
        if check(m):
            r = m - 1
        else:
            l = m + 1
    return l
```

## template 2: find maximum value

```python
def fn(arr):
    def check(x):
        # some code
        return BOOLEAN

    l = MINIMUM_POSSIBLE_ANSWER
    r = MAXIMUM_POSSIBLE_ANSWER
    while l <= r:
        m = (r - l) // 2
        if check(m):
            l = m + 1
        else:
            r = m - 1
    return r
```