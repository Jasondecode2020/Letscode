### 2047. Number of Valid Words in a Sentence

```python
class Solution:
    def countValidWords(self, sentence: str) -> int:
        res = 0
        def check(word):
            if re.match("[a-z]*([a-z]-[a-z])?[a-z]*[,.!]?$", word):
                return True
            return False

        for w in sentence.split():
            if check(w):
                res += 1
        return res
```

### 65. Valid Number

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        reg = '^[+-]?((\d+\.?)|(\d*\.\d+))([eE][+-]?\d+)?$'
        if re.match(reg, s):
            return True
        return False
```


### 465. Optimal Account Balancing

```python
class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        person = defaultdict(int)
        for u, v, c in transactions:
            person[u] -= c
            person[v] += c
        accounts = list(person.values())
       
        self.res = inf
        n = len(accounts)
        def dfs(i, cnt):
            if cnt >= self.res: return 
            while i < n and accounts[i] == 0: 
                i += 1
            if i == n:
                self.res = min(self.res, cnt)
                return
              
            for j in range(i + 1, n):
                if accounts[i] * accounts[j] < 0:
                    accounts[j] += accounts[i]
                    dfs(i + 1, cnt + 1)
                    accounts[j] -= accounts[i]
        dfs(0, 0)
        return self.res
```