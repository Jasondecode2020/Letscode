# template: build prefix sum array

```python
def fn(arr):
    n = len(arr)
    for i in range(1, n):
        arr[i] += arr[i - 1]
```

- lib

```python
def fn(arr):
    arr = list(accumulate(arr, initial = 0))
    # or 
    arr = list(accumulate(arr))
```