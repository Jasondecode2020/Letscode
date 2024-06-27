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
for _ in range(q):
  def check(threshold):
    res = 0
    for a, c in zip(attack, cooldown):
      res += ((threshold - 1) // c + 1) * a
    return res >= h #
    
  h, n = LI()
  attack, cooldown = LI(), LI()
  mx_cd = max(cooldown)
  l, r, res = 1, 2 ** 64, 1
  while l <= r:
    m = l + (r - l) // 2
    if check(m):
      res = m 
      r = m - 1
    else:
      l = m + 1
  arr.append(str(res))
print('\n'.join(arr))