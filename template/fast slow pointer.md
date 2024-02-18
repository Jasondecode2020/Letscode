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

### 2095. Delete the Middle Node of a Linked List

```python
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        slow = fast = dummy
        while fast.next and fast.next.next:
            slow = slow.next 
            fast = fast.next.next
        slow.next = slow.next.next
        return dummy.next
```