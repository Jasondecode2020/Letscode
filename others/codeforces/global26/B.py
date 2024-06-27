q = int(input())
for i in range(q):
  n = int(input())
  arr = [int(i) for i in str(n)]
  L = len(arr)
  flag = False
  for i in range(L - 1, 0, -1):
    if arr[i] == 9:
      print('NO')
      flag = True
      break
    else:
      arr[i - 1] = arr[i - 1] - 1 if arr[i - 1] >= 1 else 9
  if 0 <= int(''.join([str(i) for i in arr[:2]])) < 9:
    print('YES')
  else:
    if not flag:
      print('NO')