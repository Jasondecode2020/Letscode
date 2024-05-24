### 1 next greater

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x > arr[stack[-1]]: # next great
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### 2 next greater or equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x >= arr[stack[-1]]: # next great or equal
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### 3 next smaller

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x < arr[stack[-1]]: # next smaller
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```

### 4 next smaller or equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, x in enumerate(arr):
        while stack and x <= arr[stack[-1]]: # next smaller or equal
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```
### 5 previous greater

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] <= v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous greater index
        stack.append(i)
```

### 6 previous greater or equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] < v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous greater or equal index
        stack.append(i)
```

### 7 previous smaller

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] >= v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous smaller index
        stack.append(i)
```

### 8 previous smaller or equal

```python
def fn(arr):
    n = len(arr)
    res, stack = [0] * n, []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] > v:
            stack.pop()
        if stack:
            res[i] = stack[-1] # previous smaller or equal index
        stack.append(i)
```