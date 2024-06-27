q = int(input())
for _ in range(q):
  l, r = map(int, input().split())
  for i in range(40, -1, -1):
    if l <= 2 ** i <= r:
      print(i)
      break