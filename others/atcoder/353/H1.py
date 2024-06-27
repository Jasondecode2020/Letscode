standard_input, packages = 1, 1

if 1:
  if standard_input:
    import io, os, sys
    input = lambda: sys.stdin.readline().strip()

    import math
    inf = math.inf

    def S(): # string
      return input()
    
    def I(): # int
      return int(input())

    def II(): # iterate int from a list
      return map(int, input().split())

    def LS(): # list of string
      return list(input().split())

    def LI(): # list of int
      return list(map(int, input().split()))

    def LF(): # list of float
      return list(map(float, input().split()))

    def M10(): # iterate of 1 to 0
      return map(lambda x: int(x) - 1, input().split())

    def L10(): # list of 1 to 0
      return list(map(lambda x: int(x) - 1, input().split()))

  if packages:
    import random
    import bisect
    import typing
    from collections import Counter, defaultdict, deque
    from copy import deepcopy
    from functools import cmp_to_key, lru_cache, reduce
    from heapq import merge, heapify, heappop, heappush, heappushpop, nlargest, nsmallest
    from itertools import accumulate, combinations, permutations, count, product
    from operator import add, iand, ior, itemgetter, mul, xor
    from string import ascii_lowercase, ascii_uppercase, ascii_letters

class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def isConnected(self, n1, n2):
        return self.find(n1) == self.find(n2)

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]

q = I()
arr = []
 
def f(r, c):
  return r * C + c
directions2 = [[0, 1], [1, 0]]
 
for _ in range(q):
  R, C = LI()
  grid = [S() for _ in range(R)]
  uf = UF(R * C)
    
  for r in range(R):
    for c in range(C):
      if grid[r][c] == '#':
        for dr, dc in directions2:
          row, col = r + dr, c + dc
          if row < R and col < C and grid[row][col] == '#' and not uf.isConnected(f(r, c), f(row, col)):
            uf.union(f(r, c), f(row, col))
  
  res = 0
  vis = [0] * R * C 
  for r in range(R):
    ans = 0
    for c in range(C):
      if grid[r][c] == '.':
        ans += 1
      for dr, dc in [[-1, 0], [1,  0], [0, 0]]:
        row, col = r + dr, c + dc 
        if 0 <= row < R and 0 <= col < C and grid[row][col] == '#' and not vis[uf.find(f(row, col))]:
            vis[uf.find(f(row, col))] = 1
            ans += uf.rank[uf.find(f(row, col))]
    for c in range(C):
       for dr, dc in [[-1, 0], [1,  0], [0, 0]]:
          row, col = r + dr, c + dc 
          if 0 <= row < R and 0 <= col < C:
             vis[uf.find(f(row, col))] = 0
    res = max(res, ans)
  
  for c in range(C):
    ans = 0
    for r in range(R):
      if grid[r][c] == '.':
        ans += 1
      for dr, dc in [[0, 1], [0, -1], [0, 0]]:
        row, col = r + dr, c + dc 
        if 0 <= row < R and 0 <= col < C and grid[row][col] == '#' and not vis[uf.find(f(row, col))]:
            vis[uf.find(f(row, col))] = 1
            ans += uf.rank[uf.find(f(row, col))]
    for r in range(R):
       for dr, dc in [[0, 1], [0, -1], [0, 0]]:
          row, col = r + dr, c + dc 
          if 0 <= row < R and 0 <= col < C:
             vis[uf.find(f(row, col))] = 0
    res = max(res, ans)
  arr.append(res)
print('\n'.join(map(str, arr)))