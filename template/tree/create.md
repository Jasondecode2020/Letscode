## Create Binary Tree

* [108. Convert Sorted Array to Binary Search Tree](#108-convert-sorted-array-to-binary-search-tree)
* [109. Convert Sorted List to Binary Search Tree](#109-convert-sorted-list-to-binary-search-tree)
* [1382. Balance a Binary Search Tree](#1382-balance-a-binary-search-tree)
* [1008. Construct Binary Search Tree from Preorder Traversal](#1008-construct-binary-search-tree-from-preorder-traversal)
* [654. Maximum Binary Tree](#654-maximum-binary-tree)

* [998. Maximum Binary Tree II](#998-maximum-binary-tree-ii)
* [2196. Create Binary Tree From Descriptions](#2196-create-binary-tree-from-descriptions)
* [105. Construct Binary Tree from Preorder and Inorder Traversal](#105-construct-binary-tree-from-preorder-and-inorder-traversal)
* [106. Construct Binary Tree from Inorder and Postorder Traversal](#106-construct-binary-tree-from-inorder-and-postorder-traversal)
* [889. Construct Binary Tree from Preorder and Postorder Traversal](#889-construct-binary-tree-from-preorder-and-postorder-traversal)

* [1028. Recover a Tree From Preorder Traversal](#1028-recover-a-tree-from-preorder-traversal)
* [536. Construct Binary Tree from String](#536-construct-binary-tree-from-string)
* [1628. Design an Expression Tree With Evaluate Function](#1628-design-an-expression-tree-with-evaluate-function)
* [1597. Build Binary Expression Tree From Infix Expression](#1597-build-binary-expression-tree-from-infix-expression)

### 108. Convert Sorted Array to Binary Search Tree

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(l, r):
            if l > r:
                return None 
            m = l + (r - l) // 2
            node = TreeNode(nums[m])
            node.left = dfs(l, m - 1)
            node.right = dfs(m + 1, r)
            return node 
        return dfs(0, len(nums) - 1)
```

### 109. Convert Sorted List to Binary Search Tree

```python
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next 
            
        def dfs(l, r):
            if l > r:
                return None 
            m = l + (r - l) // 2
            node = TreeNode(nums[m])
            node.left = dfs(l, m - 1)
            node.right = dfs(m + 1, r)
            return node 
        return dfs(0, len(nums) - 1)
```

### 1382. Balance a Binary Search Tree

```python
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        nums = []
        def preorder(node):
            if node:
                preorder(node.left)
                nums.append(node.val)
                preorder(node.right)
        preorder(root)
        
        def dfs(l, r):
            if l > r:
                return None 
            m = l + (r - l) // 2
            node = TreeNode(nums[m])
            node.left = dfs(l, m - 1)
            node.right = dfs(m + 1, r)
            return node 
        return dfs(0, len(nums) - 1)
```

### 1008. Construct Binary Search Tree from Preorder Traversal

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        def check(nums):
            n = len(nums)
            res = n 
            if nums[0] == min(nums):
                res = 1 
            for i in range(1, n - 1):
                if nums[i] < nums[0] < nums[i + 1]:
                    res = i + 1
            return res 
        def dfs(nums):
            if not nums:
                return None 
            node = TreeNode(nums[0])
            i = check(nums)
            node.left = dfs(nums[1:i])
            node.right = dfs(nums[i:])
            return node 
        return dfs(preorder)
```

### 654. Maximum Binary Tree

```python
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(nums):
            if not nums:
                return None
            i = nums.index(max(nums))
            root = TreeNode(nums[i])
            root.left = dfs(nums[:i])
            root.right = dfs(nums[i + 1:])
            return root
        return dfs(nums)
```

### 998. Maximum Binary Tree II

```python
class Solution:
    def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        def dfs(node):
            if not node:
                return TreeNode(val)
            if node.val < val:
                return TreeNode(val, node, None)
            node.right = dfs(node.right)
            return node
        return dfs(root)
```

### 2196. Create Binary Tree From Descriptions

```python
class Solution:
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        d = defaultdict(TreeNode)
        vals = set()
        for parent, child, sign in descriptions:
            if sign == 1:
                d[parent].left = d[child]
            else:
                d[parent].right = d[child]
            d[parent].val = parent
            d[child].val = child 
            vals.add(child)
        return next(node for v, node in d.items() if v not in vals)
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def dfs(preorder, inorder):
            if not preorder:
                return None 
            node = TreeNode(preorder[0])
            i = inorder.index(preorder[0])
            node.left = dfs(preorder[1:i+1], inorder[:i])
            node.right = dfs(preorder[i+1:], inorder[i+1:])
            return node 
        return dfs(preorder, inorder)
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        def dfs(inorder, postorder):
            if not postorder:
                return None 
            left_size = inorder.index(postorder[-1])
            left = dfs(inorder[:left_size], postorder[:left_size])
            right = dfs(inorder[left_size + 1:], postorder[left_size: -1])
            return TreeNode(postorder[-1], left, right)
        return dfs(inorder, postorder)
```

### 889. Construct Binary Tree from Preorder and Postorder Traversal

```python
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        def dfs(preorder, postorder):
            if not preorder: 
                return None
            if len(preorder) == 1: 
                return TreeNode(preorder[0])
            left_size = postorder.index(preorder[1]) + 1
            left = dfs(preorder[1:left_size+1], postorder[:left_size])
            right = dfs(preorder[left_size+1:], postorder[left_size:-1])
            return TreeNode(preorder[0], left, right)
        return dfs(preorder, postorder)
```

### 1028. Recover a Tree From Preorder Traversal

### 536. Construct Binary Tree from String

### 1628. Design an Expression Tree With Evaluate Function

### 1597. Build Binary Expression Tree From Infix Expression
