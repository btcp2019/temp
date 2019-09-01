n, k = map(int, input().split())
a = [list(map(int, input().split())) for i in range(n)]
m = max(r for l, r in a)
dp = [1 for i in range(m + 1)]
for i in range(1, m + 1):
	dp[i] = dp[i - 1]
	if i >= k:
		dp[i] += dp[i - k]
for i in range(1, m + 1):
	dp[i] += dp[i - 1]
for l, r in a:
	print(dp[r] - dp[l - 1])