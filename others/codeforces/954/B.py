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
directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
for _ in range(q):
  R, C = LI()
  grid = [LI() for _ in range(R)]
  for r in range(R):
    for c in range(C):
      cnt, target, val = 0, 0, -inf
      for dr, dc in directions:
        row, col = r + dr, c + dc 
        if 0 <= row < R and 0 <= col < C:
          cnt += 1
          if grid[r][c] > grid[row][col]:
            target += 1
            val = max(val, grid[row][col])
      if cnt == target:
        grid[r][c] = val
  for row in grid:
    arr.append(' '.join([str(r) for r in row]))
print('\n'.join(arr))
