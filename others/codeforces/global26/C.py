q = int(input())
for i in range(q):
  n = int(input())
  arr = [int(i) for i in input().split()]
  L = len(arr)
  mn, mx = 0, 0
  for i in range(L):
    temp = mn
    mn = mn + arr[i]
    mx = max(abs(temp + arr[i]), abs(mx + arr[i]))
  print(mx)