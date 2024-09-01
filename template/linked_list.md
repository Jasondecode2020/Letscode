# linked list

## 1. Iteration(7)

* [1290. Convert Binary Number in a Linked List to Integer](#1290-convert-binary-number-in-a-linked-list-to-integer)
* [2058. Find the Minimum and Maximum Number of Nodes Between Critical Points](#2058-find-the-minimum-and-maximum-number-of-nodes-between-critical-points)
* [2181. Merge Nodes in Between Zeros](#2181-merge-nodes-in-between-zeros)
* [725. Split Linked List in Parts](#725-split-linked-list-in-parts)
* [817. Linked List Components](#817-linked-list-components)
* [3062. Winner of the Linked List Game](#2-Add-Two-Numbers)
* [3063. Linked List Frequency](#3063-linked-list-frequency)

## 2. Deletion(8)

* [203. Remove Linked List Elements](#203-remove-linked-list-elements)
* [3217. Delete Nodes From Linked List Present in Array](#3217-delete-nodes-from-linked-list-present-in-array)
* [83. Remove Duplicates from Sorted List](#2181-merge-nodes-in-between-zeros)
* [82. Remove Duplicates from Sorted List II](#82-remove-duplicates-from-sorted-list-ii)
* [237. Delete Node in a Linked List](#237-delete-node-in-a-linked-list)
* [1836. Remove Duplicates From an Unsorted Linked List](#1836-remove-duplicates-from-an-unsorted-linked-list)
* [1669. Merge In Between Linked Lists](#1669-merge-in-between-linked-lists)
* [2487. Remove Nodes From Linked List](#2487-remove-nodes-from-linked-list)

## 3. Insertion(4)

* [2807. Insert Greatest Common Divisors in Linked List](#2807-insert-greatest-common-divisors-in-linked-list)
* [2046. Sort Linked List Already Sorted Using Absolute Values](#2046-sort-linked-list-already-sorted-using-absolute-values)
* [147. Insertion Sort List](#147-insertion-sort-list)
* [708. Insert into a Sorted Circular Linked List](#708-insert-into-a-sorted-circular-linked-list)

## 4. Reversion(5)

* [206. Reverse Linked List](#206-reverse-linked-list)
* [92. Reverse Linked List II](#92-reverse-linked-list-ii)
* [24. Swap Nodes in Pairs](#24-swap-nodes-in-pairs)
* [25. Reverse Nodes in k-Group](#708-insert-into-a-sorted-circular-linked-list)
* [2074. Reverse Nodes in Even Length Groups](#2074-reverse-nodes-in-even-length-groups)

## 5. before after(4)

* [19. Remove Nth Node From End of List](#19-remove-nth-node-from-end-of-list)
* [61. Rotate List](#61-rotate-list)
* [1721. Swapping Nodes in a Linked List](#1721-swapping-nodes-in-a-linked-list)
* [1474. Delete N Nodes After M Nodes of a Linked List](#1474-delete-n-nodes-after-m-nodes-of-a-linked-list)

## 6. slow faster pointers(8)

* [876. Middle of the Linked List](#876-middle-of-the-linked-list)
* [2095. Delete the Middle Node of a Linked List](#2095-delete-the-middle-node-of-a-linked-list)
* [141. Linked List Cycle](#141-linked-list-cycle)
* [142. Linked List Cycle II](#142-linked-list-cycle-ii)
* [234. Palindrome Linked List](#234-palindrome-linked-list)
* [2130. Maximum Twin Sum of a Linked List](#2130-maximum-twin-sum-of-a-linked-list)
* [143. Reorder List](#143-reorder-list)
* [2674. Split a Circular Linked List](#2674-split-a-circular-linked-list)

## 7. two pointers(3)

* [160. Intersection of Two Linked Lists](#160-intersection-of-two-linked-lists)
* [86. Partition List](#86-partition-list)
* [328. Odd Even Linked List](#328-odd-even-linked-list)

## 8. merge(6)

* [2. Add Two Numbers](#2-Add-Two-Numbers)
* [445. Add Two Numbers II](#445-add-two-numbers-ii)
* [2816. Double a Number Represented as a Linked List](#2816-double-a-number-represented-as-a-linked-list)
* [21. Merge Two Sorted Lists](#21-merge-two-sorted-lists)
* [369. Plus One Linked List](#369-plus-one-linked-list)
* [1634. Add Two Polynomials Represented as Linked Lists]()

## 9. d & q

* [23. Merge k Sorted Lists](#23-merge-k-sorted-lists)
* [148. Sort List](#148-sort-list)

## 10. complicated

* [707. Design Linked List](#707-design-linked-list)
* [1171. Remove Zero Sum Consecutive Nodes from Linked List]()
* [146. LRU Cache](#146-lru-cache)
* [460. LFU Cache](#460-lfu-cache)
* [432. All O`one Data Structure]
* [1206. Design Skiplist]


## 11. others

* [138. Copy List with Random Pointer](#1836-remove-duplicates-from-an-unsorted-linked-list)
* [382. Linked List Random Node]
* [430. Flatten a Multilevel Doubly Linked List]
* [1265. Print Immutable Linked List in Reverse]

### 1290. Convert Binary Number in a Linked List to Integer

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        res = 0
        while head:
            res = res * 2 + head.val
            head = head.next 
        return res
```

### 2058. Find the Minimum and Maximum Number of Nodes Between Critical Points

```python
class Solution:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        res = []
        while head:
            res.append(head.val)
            head = head.next 
        ans = []
        for i in range(1, len(res) - 1):
            if (res[i] > res[i - 1] and res[i] > res[i + 1]) or (res[i] < res[i - 1] and res[i] < res[i + 1]):
                ans.append(i)
        if len(ans) <= 1:
            return [-1, -1]
        mn = min(b - a for a, b in pairwise(ans))
        mx = ans[-1] - ans[0]
        return [mn, mx]
```

### 2181. Merge Nodes in Between Zeros

```python
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = p = ListNode()
        total = 0
        head = head.next 
        while head:
            if head.val == 0:
                p.next = ListNode(total)
                p = p.next 
                total = 0
            total += head.val 
            head = head.next 
        return dummy.next 
```

### 725. Split Linked List in Parts

```python
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        count = 0
        p = head
        while head:
            count += 1
            head = head.next 

        d, m = divmod(count, k)
        cnt = [d + 1] * m  + [d] * (k - m)
        ans = []
        for c in cnt:
            ans.append(p)
            for i in range(c):
                prev = p
                p = p.next 
            if c:
                prev.next = None
        return ans
```

### 817. Linked List Components

```python
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        s = set(nums)
        res = 0
        flag = True
        while head:
            if head.val not in s:
                flag = True
            elif flag:
                flag = False
                res += 1
            head = head.next 
        return res 
```

### 3062. Winner of the Linked List Game

```python
class Solution:
    def gameResult(self, head: Optional[ListNode]) -> str:
        even, odd = 0, 0
        while head and head.next:
            if head.val > head.next.val:
                even += 1
            elif head.val < head.next.val:
                odd += 1
            head = head.next.next 
        if even == odd:
            return 'Tie'
        return 'Odd' if even < odd else 'Even'
```

### 3063. Linked List Frequency

```python
class Solution:
    def frequenciesOfElements(self, head: Optional[ListNode]) -> Optional[ListNode]:
        d = defaultdict(int)
        p = dummy = ListNode()
        while head:
            d[head.val] += 1
            head = head.next 
        for v in d.values():
            p.next = ListNode(v)
            p = p.next 
        return dummy.next 
```

### 237. Delete Node in a Linked List

```python
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val 
        node.next = node.next.next 
```

### 83. Remove Duplicates from Sorted List

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
      p = head
      while p and p.next:
        if p.val == p.next.val:
          p.next = p.next.next
        else:
          p = p.next 
      return head
```

### 82. Remove Duplicates from Sorted List II

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = p = ListNode(next = head)
        while p.next and p.next.next:
            val = p.next.val 
            if val == p.next.next.val:
                while p.next and p.next.val == val:
                    p.next = p.next.next 
            else:
                p = p.next 
        return dummy.next 
```

### 203. Remove Linked List Elements

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = p = ListNode(next=head)
        while p and p.next:
            if p.next.val == val:
                p.next = p.next.next
            else: 
                p = p.next
        return dummy.next 
```

### 3217. Delete Nodes From Linked List Present in Array

```python
class Solution:
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = p = ListNode(next=head)
        s = set(nums)
        while p and p.next:
            if p.next.val in s:
                p.next = p.next.next
            else:
                p = p.next 
        return dummy.next 
```

### 1836. Remove Duplicates From an Unsorted Linked List

```python
class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        d = defaultdict(int)
        p = dummy = ListNode(next=head)
        while head:
            d[head.val] += 1
            head = head.next

        while p and p.next:
            if d[p.next.val] > 1:
                p.next = p.next.next
            else:
                p = p.next
        return dummy.next
```

### 1669. Merge In Between Linked Lists

```python
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        dummy = ListNode()
        p = dummy
        dummy.next = list1 
        cnt = 0
        while p:
            cnt += 1
            p = p.next 
            if cnt == a:
                A = p 
            if cnt == b + 2:
                B = p 
                break 
        p = list2 
        while p.next:
            p = p.next 
        A.next = list2 
        p.next = B 
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
        for n in stack:
            p.next = ListNode(n)
            p = p.next 
        return dummy.next 
```

### 2807. Insert Greatest Common Divisors in Linked List

```python
class Solution:
    def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur.next:
            cur.next = ListNode(gcd(cur.val, cur.next.val), cur.next)
            cur = cur.next.next
        return head
```

### 2046. Sort Linked List Already Sorted Using Absolute Values

```python
class Solution:
    def sortLinkedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # find positive and negative linked list
        pos, neg = [], []
        while head:
            if head.val >= 0:
                pos.append(head.val)
            else:
                neg.append(head.val)
            head = head.next
        # connect array
        neg.reverse()
        res = neg + pos
        dummy = p = ListNode()
        for i in range(len(res)):
            p.next = ListNode(res[i])
            p = p.next 
        return dummy.next
```

### 147. Insertion Sort List

```python
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        prev, cur = head, head.next 
        while cur:
            if cur.val >= prev.val:
                prev, cur = cur, cur.next
                continue 
            p = dummy 
            while cur.val > p.next.val:
                p = p.next 
            prev.next = cur.next 
            cur.next = p.next 
            p.next = cur 
            cur = prev.next 
        return dummy.next 
```

### 708. Insert into a Sorted Circular Linked List

```python
class Solution:
    def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
        node = Node(val=insertVal)
        if not head:
            node.next = node
            return node
        curr = head.next
        while curr != head:
            if curr.val <= insertVal <= curr.next.val:
                break
            if curr.val > curr.next.val:
                if insertVal >= curr.val or insertVal <= curr.next.val:
                    break
            curr = curr.next
        node.next = curr.next
        curr.next = node
        return head
```

### 206. Reverse Linked List

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head
        while cur:
            nxt = cur.next 
            cur.next = prev 
            prev, cur = cur, nxt 
        return prev
```

### 92. Reverse Linked List II

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = p = ListNode(next=head)
        for _ in range(left - 1):
            p = p.next
        
        prev, cur = None, p.next 
        for _ in range(right - left + 1):
            nxt = cur.next 
            cur.next = prev 
            prev, cur = cur, nxt
        p.next.next = cur 
        p.next = prev
        return dummy.next 
```

### 24. Swap Nodes in Pairs

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # prev -> 1 -> 2 -> 3 -> 4
        dummy = ListNode()
        dummy.next = head
        prev, cur = dummy, head

        total = 0
        while head:
            total += 1
            head = head.next 

        def swapOnePair():
            nxt = cur.next
            cur.next = nxt.next 
            nxt.next = prev.next
            prev.next = nxt 

        for i in range(total // 2):
            swapOnePair()
            prev, cur = cur, cur.next 
        return dummy.next
```

### 25. Reverse Nodes in k-Group

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        prev, cur = dummy, head

        total = 0
        while head:
            total += 1
            head = head.next 

        def swapOnePair():
            nxt = cur.next
            cur.next = nxt.next 
            nxt.next = prev.next
            prev.next = nxt 

        for i in range(total // k):
            for j in range(k - 1):
                swapOnePair()
            prev, cur = cur, cur.next 
        return dummy.next
```

### 2074. Reverse Nodes in Even Length Groups

```python
class Solution:
    def reverseEvenLengthGroups(self, head: Optional[ListNode]) -> Optional[ListNode]:
        res = []
        val = 1
        i = 0
        ans = []
        while head:
            ans.append(head.val)
            i += 1
            if i == val:
                if val % 2 == 0:
                    res.extend(ans[::-1])
                else:
                    res.extend(ans)
                i = 0
                val = val + 1
                ans = []
            head = head.next 
        if ans:
            if len(ans) % 2 == 0:
                res.extend(ans[::-1])
            else:
                res.extend(ans)
        dummy = p = ListNode()
        for n in res:
            p.next = ListNode(n)
            p = p.next 
        return dummy.next 
```

### 19. Remove Nth Node From End of List

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head 
        slow = fast = dummy
        for i in range(n + 1):
            fast = fast.next 
        while fast:
            fast = fast.next
            slow = slow.next 
        slow.next = slow.next.next 
        return dummy.next 
```

### 61. Rotate List

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return None
        after = dummy = ListNode(next=head)
        total = 0
        while head:
            total += 1
            head = head.next 
        for _ in range(k % total):
            after = after.next 
        before = dummy 
        while after.next:
            before = before.next 
            after = after.next 
        nxt = before.next 
        if not nxt:
            return dummy.next
        before.next = None 
        after.next = dummy.next 
        return nxt
```

### 1721. Swapping Nodes in a Linked List

```python
```

### 876. Middle of the Linked List

```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head 
        while fast and fast.next:
            slow = slow.next 
            fast = fast.next.next 
        return slow
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

### 141. Linked List Cycle

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head 
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
            if slow == fast:
                return True
        return False
```

### 142. Linked List Cycle II

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head 
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
            if slow == fast:
                while slow != head:
                    head = head.next 
                    slow = slow.next 
                return head
        return None
```

### 234. Palindrome Linked List

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        total = 0
        p = head 
        while p:
            total += 1
            p = p.next 

        p = dummy = ListNode()
        p.next = head 
        slow = fast = p
        while fast and fast.next:
            slow = slow.next 
            fast = fast.next.next 
        if total % 2 == 0:
            l2 = slow.next 
        else:
            l2 = slow 

        def reverseLinkedList(prev, cur):
            while cur:
                nxt = cur.next 
                cur.next = prev 
                prev, cur = cur, nxt 
            return prev 

        p2 = reverseLinkedList(None, l2)
        while head and p2:
            if head.val != p2.val:
                return False
            head = head.next 
            p2 = p2.next 
        return True
```

### 2130. Maximum Twin Sum of a Linked List

```python
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        arr = []
        while head:
            arr.append(head.val)
            head = head.next 
        l, r = 0, len(arr) - 1
        res = 0
        while l < r:
            res = max(res, arr[l] + arr[r])
            l += 1
            r -= 1
        return res
```

### 143. Reorder List

```python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # find the mid node
        slow = fast = ListNode()
        slow.next = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        p = slow.next
        slow.next = None
        # reverse
        def reverseList(prev, cur):
            if not cur:
                return prev
            nxt = cur.next
            cur.next = prev
            return reverseList(cur, nxt)
        tail = reverseList(None, p)
        # merge
        l1, l2 = head, tail
        while l1 and l2:
            l1_temp, l2_temp = l1.next, l2.next 
            l1.next = l2 
            l2.next = l1_temp 
            l1, l2 = l1_temp, l2_temp
```

### 2674. Split a Circular Linked List

```python
class Solution:
    def splitCircularLinkedList(self, list: Optional[ListNode]) -> List[Optional[ListNode]]:
        slow, fast = list, list.next 
        while fast != list and fast.next != list:
            slow = slow.next 
            fast = fast.next 
            if fast.next != list:
                fast = fast.next 
        list2 = slow.next 
        fast.next = list2 
        slow.next = list 
        return [list, list2]
```

### 160. Intersection of Two Linked Lists

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        h1, h2 = headA, headB
        while h1 is not h2:
            h1 = h1.next if h1 else headB
            h2 = h2.next if h2 else headA
        return h1
```

### 86. Partition List

```python
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        # 2 pass
        r = dummy = ListNode(0)
        p = q = head
        while p:
            if p.val < x:
                r.next = ListNode(p.val)
                r = r.next
            p = p.next
        while q:
            if q.val >= x:
                r.next = ListNode(q.val)
                r = r.next
            q = q.next
        return dummy.next
```

### 328. Odd Even Linked List

```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        # use 2 list, one for odd, one for even
        odd = oddHead = head
        even = evenHead = odd.next
        while even and even.next:
            odd.next = even.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        odd.next = evenHead
        return head
```

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

### 445. Add Two Numbers II

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(head):
            prev, curr = None, head
            while curr:
                next = curr.next
                curr.next = prev
                prev = curr
                curr = next
            return prev
        # turn to leetcode 2
        lst1 = reverse(l1)
        lst2 = reverse(l2)
        temp = dummy = ListNode(0)
        carry = 0
        while lst1 and lst2:
            val = lst1.val + lst2.val + carry
            temp.next = ListNode(val % 10)
            lst1 = lst1.next
            lst2 = lst2.next
            temp = temp.next
            if val >= 10:
                carry = 1
            else:
                carry = 0
        while lst1 and not lst2:
            val = lst1.val + carry
            temp.next = ListNode(val % 10)
            lst1 = lst1.next
            temp = temp.next
            if val >= 10:
                carry = 1
            else:
                carry = 0
        while lst2 and not lst1:
            val = lst2.val + carry
            temp.next = ListNode(val % 10)
            lst2 = lst2.next
            temp = temp.next
            if val >= 10:
                carry = 1
            else:
                carry = 0
        if carry == 1:
            temp.next = ListNode(1)
            
        return reverse(dummy.next)
```

### 2816. Double a Number Represented as a Linked List

```python
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # head = [1,8,9]
        res1 = []
        while head:
            res1.append(head.val)
            head = head.next
        res2 = res1[::]
        carry = 0
        dummy = p = ListNode()
        i = len(res1) - 1
        res = deque()
        while i >= 0:
            val = res1[i] * 2 + carry
            res.appendleft(val % 10)
            carry = val // 10
            i -= 1
        if carry: res.appendleft(1)
        dummy = p = ListNode()
        for n in res:
            p.next = ListNode(n)
            p = p.next
        return dummy.next
```

### 21. Merge Two Sorted Lists

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        p = dummy = ListNode()
        while list1 and list2:
            if list1.val > list2.val:
                p.next = ListNode(list2.val)
                list2 = list2.next 
            else:
                p.next = ListNode(list1.val)
                list1 = list1.next 
            p = p.next 
        p.next = list1 if list1 else list2
        return dummy.next 
```

### 369. Plus One Linked List

```python
class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        # 1 -> 2 -> 3
        def reverseList(prev, cur):
            if not cur:
                return prev
            nxt = cur.next
            cur.next = prev
            return reverseList(cur, nxt)
        p = reverseList(None, head)
        # 3 -> 2 -> 1
        carry = 1
        head = p
        while p:
            value = p.val + carry
            p.val = (value) % 10
            carry = value // 10
            if p.next:
                p = p.next
            else:
                break
        # 4 -> 2 -> 1
        if carry:
            p.next = ListNode(1)
        # 1 -> 2 -> 4
        return  reverseList(None, head)
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

### 23. Merge k Sorted Lists

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        p = dummy = ListNode()
        pq = []
        for l in lists:
            while l:
                heappush(pq, l.val)
                l = l.next 
        while pq:
            val = heappop(pq)
            p.next = ListNode(val)
            p = p.next
        return dummy.next
```

### 148. Sort List

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def merge(left, right):
            p = dummy = ListNode()
            while left and right:
                if left.val < right.val:
                    p.next = left
                    left = left.next
                else:
                    p.next = right
                    right = right.next
                p = p.next
            p.next = left if left else right
            return dummy.next

        def merge_sort(head):
            if not head or not head.next:
                return head 
            slow, fast = head, head.next 
            while fast and fast.next:
                fast, slow = fast.next.next, slow.next 
            mid, slow.next = slow.next, None
            left, right = merge_sort(head), merge_sort(mid)
            return merge(left, right)
        return merge_sort(head)
```

### 1019. Next Greater Node In Linked List

```python
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next 

        n = len(nums)
        res, stack = [0] * n, []
        for i, x in enumerate(nums):
            while stack and x > nums[stack[-1]]:
                j = stack.pop()
                res[j] = x 
            stack.append(i)
        return res 
```

### 707. Design Linked List

```python
class ListNode:
    def __init__(self, val = 0):
        self.val = val 
        self.next = None

class MyLinkedList:

    def __init__(self):
        self.head = ListNode()
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        p = self.head 
        for i in range(index + 1):
            p = p.next 
        return p.val 

    def addAtHead(self, val: int) -> None:
        return self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        return self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return 
        self.size += 1
        p = self.head 
        for i in range(index):
            p = p.next 
        temp = ListNode(val)
        temp.next = p.next 
        p.next = temp

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return 
        self.size -= 1
        p = self.head 
        for i in range(index):
            p = p.next 
        p.next = p.next.next 
```

### 146. LRU Cache

```python
class ListNode:
    def __init__(self, key = 0, value = 0):
        self.key = key
        self.value = value

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.head = self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = ListNode(key, value)
            self.cache[key] = node
            self.addToHead(node)
            self.capacity -= 1
            if self.capacity < 0:
                removed = self.removeTail(self.tail.prev)
                self.cache.pop(removed.key)
        else:
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeTail(self, node):
        self.removeNode(node)
        return node
```

### 460. LFU Cache

```python
class Node:
    def __init__(self, key, val, pre=None, nex=None, freq=0):
        self.pre = pre
        self.nex = nex
        self.freq = freq
        self.val = val
        self.key = key
        
    def insert(self, nex):
        nex.pre = self
        nex.nex = self.nex
        self.nex.pre = nex
        self.nex = nex
    
def create_linked_list():
    head = Node(0, 0)
    tail = Node(0, 0)
    head.nex = tail
    tail.pre = head
    return (head, tail)

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.minFreq = 0
        self.freqMap = collections.defaultdict(create_linked_list)
        self.keyMap = {}

    def delete(self, node):
        if node.pre:
            node.pre.nex = node.nex
            node.nex.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.nex is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
        return node.key
        
    def increase(self, node):
        node.freq += 1
        self.delete(node)
        self.freqMap[node.freq][-1].pre.insert(node)
        if node.freq == 1:
            self.minFreq = 1
        elif self.minFreq == node.freq - 1:
            head, tail = self.freqMap[node.freq - 1]
            if head.nex is tail:
                self.minFreq = node.freq

    def get(self, key: int) -> int:
        if key in self.keyMap:
            self.increase(self.keyMap[key])
            return self.keyMap[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity != 0:
            if key in self.keyMap:
                node = self.keyMap[key]
                node.val = value
            else:
                node = Node(key, value)
                self.keyMap[key] = node
                self.size += 1
            if self.size > self.capacity:
                self.size -= 1
                deleted = self.delete(self.freqMap[self.minFreq][0].nex)
                self.keyMap.pop(deleted)
            self.increase(node)
```