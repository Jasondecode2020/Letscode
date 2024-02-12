## linked list

* [2. Add Two Numbers](#2-Add-Two-Numbers)

### 2. Add Two Numbers

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = p = ListNode()
        carry = 0
        while l1 or l2:
            val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            p.next = ListNode(val % 10)
            carry = val // 10
            p = p.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if carry:
            p.next = ListNode(1)
        return dummy.next
```