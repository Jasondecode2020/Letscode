## template

```python
'''
Leetcode 14
'''
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool: # check if a word inside trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def startsWith(self, prefix: str) -> bool: # check if a prefix inside a word
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
        
class Solution:
    def fn(self, strs: List[str]) -> str:
        t = Trie()
        for word in strs:
            t.insert(word) # insert all words to build trie for search
            
        res = ''
        cur = t.root
        while cur:
            if len(cur.children) > 1 or cur.endOfWord: # if a word ends or has more than one children
                break
            c = list(cur.children.keys())[0]
            cur = cur.children[c]
            res += c
        return res
```

* [14. Longest Common Prefix](#14-Longest-Common-Prefix)
* [208. Implement Trie](#208-Implement-Trie)
* [720. Longest Word in Dictionary](#720-Longest-Word-in-Dictionary)
* [1804. Implement Trie II](#1804-Implement-Trie-II)

### 14. Longest Common Prefix

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        t = Trie()
        for word in strs:
            t.insert(word)

        res = ''
        cur = t.root
        while cur:
            if len(cur.children) > 1 or cur.endOfWord: # if a word ends or has more than one children
                break
            c = list(cur.children.keys())[0]
            cur = cur.children[c]
            res += c
        return res
```

### 208. Implement Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```

### 720. Longest Word in Dictionary

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):
        cur = self.root 
        for c in w:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, w):
        cur = self.root 
        for c in w:
            cur = cur.children[c]
            if not cur.endOfWord:
                return False
        return True

class Solution:
    def longestWord(self, words: List[str]) -> str:
        t = Trie()
        for w in words:
            t.insert(w)
        res = []
        for w in words:
            if t.search(w):
                res.append(w)

        if not res:
            return ''
        res.sort(key = lambda x: (-len(x), x))
        return res[0]
```

### 1804. Implement Trie II

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count_word = 0
        self.count_prefix = 0

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.count_prefix += 1
        cur.count_word += 1

    def countWordsEqualTo(self, word: str) -> int:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return 0
            cur = cur.children[c]
        return cur.count_word

    def countWordsStartingWith(self, prefix: str) -> int:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return 0
            cur = cur.children[c]
        return cur.count_prefix

    def erase(self, word: str) -> None:
        cur = self.root
        for c in word:
            cur = cur.children[c]
            cur.count_prefix -= 1
        cur.count_word -= 1
```

### 648. Replace Words

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        t = Trie()
        for word in dictionary:
            t.insert(word)

        words = sentence.split(' ')
        dictionary = set(dictionary)
        for i, w in enumerate(words):
            for j in range(len(w)):
                if t.search(w[:j+1]):
                    words[i] = w[:j+1]
                    break
        return ' '.join(words)
```

- set

```python
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        words = sentence.split(' ')
        dictionary = set(dictionary)
        for i, w in enumerate(words):
            for j in range(len(w)):
                if w[:j+1] in dictionary:
                    words[i] = w[:j+1]
                    break
        return ' '.join(words)
```

### 211. Design Add and Search Words Data Structure

- brute force

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True
    
    def find(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def search(self, word: str) -> bool:
        word_set = ['']
        for c in word:
            if c == '.':
                temp = []
                for w in word_set:
                    for letter in ascii_lowercase:
                        temp.append(w + letter)
                word_set = temp
            else:
                word_set = [w + c for w in word_set]
        for w in word_set:
            if self.find(w):
                return True
        return False
```

- dfs

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        def find(idx, root):
            if not root:
                return False
            if idx == len(word):
                return root.endOfWord
            if word[idx] != '.':
                return root and find(idx + 1, root.children.get(word[idx], None))
            else:
                for child in root.children.values():
                    if find(idx + 1, child):
                        return True
            return False
        return find(0, self.root)
```

### 212. Word Search II

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        t = Trie()
        words = set(words)
        for word in words:
            t.insert(word)
        
        def dfs(r, c, node, path):
            char = board[r][c]
            if char not in node.children:
                return
            child = node.children[char]
            if child.endOfWord:
                res.add(path + char)
                if not child.children:
                    return

            board[r][c] = "#"
            for dr, dc in directions:
                row, col = r + dr, c + dc
                if 0 <= row < R and 0 <= col < C:
                    dfs(row, col, child, path + char)
            board[r][c] = char

        R, C = len(board), len(board[0])
        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        res = set()
        for r in range(R):
            for c in range(C):
                dfs(r, c, t.root, '')
        return list(res)
```

### 676. Implement Magic Dictionary

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class MagicDictionary:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:
            self.insert(word) # insert all words to build trie for search

    def search(self, searchWord: str) -> bool:
        def dfs(idx, root, count):
            if idx == len(searchWord):
                if count == 1:
                    return root.endOfWord
                return False
            ch = searchWord[idx]
            if ch in root.children:
                if dfs(idx + 1, root.children[ch], count):
                    return True
            if count == 0:
                for child in root.children:
                    if child != ch:
                        if dfs(idx + 1, root.children[child], count + 1):
                            return True
            return False
        return dfs(0, self.root, 0)
```

### 820. Short Encoding of Words

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool: # check if a word inside trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

    def startsWith(self, prefix: str) -> bool: # check if a prefix inside a word
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True

    def tialOfWord(self, word):
        cur = self.root
        for c in word:
            cur = cur.children[c]
        return len(cur.children) == 0

class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        t = Trie()
        res = 0
        words = set(words)
        for word in words:
            t.insert(word[::-1])
        for word in words:
            if t.tialOfWord(word[::-1]):
                res += len(word) + 1
        return res 
```

### 1032. Stream of Characters

```python
class TrieNode: # has children and endOfWord
    def __init__(self):
        self.children = {} # can be 26 for English letters
        self.endOfWord = False # check if word ends with a letter

class Trie: # a tree like data structure to solve prefix problems in string
    def __init__(self): # init the node
        self.root = TrieNode()

    def insert(self, word: str) -> None: # insert a word inside a Trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool: # check if a word inside trie
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord == True

class StreamChecker:

    def __init__(self, words: List[str]):
        self.words = set(words)
        self.t = Trie()
        self.s = ''
        for word in words:
            self.t.insert(word[::-1])

    def query(self, letter: str) -> bool:
        self.s += letter
        if len(self.s) > 200:
            reversed_str = self.s[-200:][::-1]
        else:
            reversed_str = self.s[::-1]
        for i in range(1, len(reversed_str) + 1):
            if self.t.search(reversed_str[:i]):
                return True
        return False



# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)
```

### 745. Prefix and Suffix Search

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        self.index = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, i):
        cur = self.root 
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
                cur.index.add(i)
            cur = cur.children[c]
            cur.index.add(i)
        cur.endOfWord = True 

    def prefix(self, word):
        cur = self.root 
        for c in word:
            if c not in cur.children:
                return -1
            cur = cur.children[c]
        return cur.index

class WordFilter:

    def __init__(self, words: List[str]):
        self.t1, self.t2 = Trie(), Trie()
        d = defaultdict(int)
        for i, word in enumerate(words):
            d[word] = i
        for word in d.keys():
            self.t1.insert(word, d[word])
            self.t2.insert(word[::-1], d[word])

    def f(self, pref: str, suff: str) -> int:
        preIndexSet, sufIndexSet = self.t1.prefix(pref), self.t2.prefix(suff[::-1])
        if preIndexSet == -1 or sufIndexSet == -1:
            return -1
        res = -inf
        for cnt in preIndexSet:
            if cnt in sufIndexSet:
                res = max(res, cnt)
        return res if res != -inf else -1
        

# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(pref,suff)
```