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

### 1474. Delete N Nodes After M Nodes of a Linked List

```python
class Solution:
    def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        count1, count2 = 0, 0
        p = dummy = ListNode()
        p.next = head 
        while p.next:
            count1 += 1
            p = p.next 
            if count1 == m:
                count1 = 0
                while p.next:
                    p.next = p.next.next 
                    count2 += 1
                    if count2 == n:
                        count2 = 0
                        break
        return dummy.next
```

### 2487. Remove Nodes From Linked List

```python
class Solution:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        stack = []
        while head:
            val = head.val
            while stack and val > stack[-1]:
                stack.pop()
            stack.append(val)
            head = head.next
        dummy = p = ListNode()
        for i in range(len(stack)):
            p.next = ListNode(stack[i])
            p = p.next 
        return dummy.next 
```