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

q = I()
arr = []
directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
for _ in range(q):
  n = I()
  grid = []
  for _ in range(2):
    grid.append(S())
  q = deque([(0, 0)])
  visited = set([(0, 0)])
  flag = False
  while q:
    r, c = q.popleft()
    if r == 1 and c == n - 1:
      flag = True
      arr.append('YES')
    if (r + c) % 2 == 0:
      for dr, dc in directions:
        row, col = r + dr, c + dc
        if 0 <= row < 2 and 0 <= col < n and (row, col) not in visited:
          q.append((row, col))
          visited.add((row, col))
    else:
      if grid[r][c] == '>' and c + 1 < n and (r, c + 1) not in visited:
        q.append((r, c + 1))
        visited.add((r, c + 1))
      elif grid[r][c] == '<' and c - 1 >= 0 and (r, c - 1) not in visited:
        q.append((r, c - 1))
        visited.add((r, c - 1))
  if not flag:
    arr.append('NO')
print('\n'.join(arr))