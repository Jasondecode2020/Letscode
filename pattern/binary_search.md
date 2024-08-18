```python
def lower_bound(nums, target):
    l, r = 0, n - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] < target:
            l = m + 1
        else:
            r = m - 1
    return l
```

```python
bisect_left(nums, target)
```