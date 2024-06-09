```python
def fn(arr):
    l, res = 0, 0
    for r, v in enumerate(arr):
        # may need some code
        while WINDOW_BROKEN:
            # handle l pointer of window
            l += 1
        res = max(res, r - l + 1)
    return res
```