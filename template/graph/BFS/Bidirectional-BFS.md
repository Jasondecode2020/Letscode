## template: bidirectional bfs

## question list

* `127. Word Ladder`
* 433
* `752. Open the Lock`
* 934
* 1036
* 332
* 1091
* 1654

### 752. Open the Lock

- normal bfs 

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        def check(s):
            d, res = defaultdict(list), []
            for c in [str(i) for i in range(0, 10)]:
                s1, s2 = '0', '9'
                if c != '9':
                    s1 = str(int(c) + 1)
                if c != '0':
                    s2 = str(int(c) - 1)
                d[c].extend([s1, s2])
            for i, c1 in enumerate(s):
                for c2 in d[c1]:
                    res.append(s[:i] + c2 + s[i + 1:])
            return res

        q = deque([('0000', 0)])
        visited = set(deadends)
        if '0000' in visited:
            return -1
        while q:
            s, cost = q.popleft()
            if s == target:
                return cost
            for child in check(s):
                if child not in visited:
                    visited.add(child)
                    q.append((child, cost + 1))
        return -1
```

### 127. Word Ladder

- normal bfs 

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        def check(word):
            res = []
            for i, c in enumerate(word):
                for l in ascii_lowercase:
                    if l != c:
                        res.append(word[:i] + l + word[i + 1:])
            return res

        wordList = set(wordList)
        q = deque([(beginWord, 1)])
        visited = set([beginWord])
        while q:
            begin, cost = q.popleft()
            if begin == endWord:
                return cost
            for word in check(begin):
                if word not in visited and word in wordList:
                    visited.add(word)
                    q.append((word, cost + 1))
        return 0
```

- bidirectional bfs

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordDict = set(wordList)
        if endWord not in wordDict: 
            return 0
        l, s1, s2 = len(beginWord), {beginWord}, {endWord}
        wordDict.remove(endWord)
        step = 1
        while s1 and s2:
            step += 1
            if len(s1) > len(s2): s1, s2 = s2, s1
            s = set()   
            for w in s1:
                new_words = [w[:i] + t + w[i+1:] for t in string.ascii_lowercase for i in range(l)]
                for new_word in new_words:
                    if new_word in s2: 
                        return step
                    if new_word in wordDict: 
                        wordDict.remove(new_word)                        
                        s.add(new_word)
            s1 = s
        return 0
```