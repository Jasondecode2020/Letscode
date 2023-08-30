## template

```python
def fn(head):
    slow, fast, res = head, head, 0
    while fast and fast.next:
        # according to problem
        slow = slow.next
        fast = fast.next.next
    return res
```