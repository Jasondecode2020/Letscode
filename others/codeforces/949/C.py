n = 2 * 10 ** 5 + 1
dp = [1] * n
for i in range(2, n):
    dp[i] = dp[i - 1] + sum(int(j) for j in str(i))
q = int(input())
for i in range(q):
  m = int(input())
  res = dp[m]
  print(res)