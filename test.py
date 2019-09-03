'''
dp[l][r]表示把[l,r]这个区间全都消除的最小次数
那么就有转移方程
dp[l][r] = min(dp[l][k] + dp[k + 1][r], dp[l][r]), k属于[l, r)
表示将当前区间分成左右两部分，先消除一边再消除一边，取最小
if (a[l] == a[r]) dp[l][r] = min(dp[l][r], dp[l + 1][r - 1])
如果两端元素相同，那么可以在消除[l+1,r-1]这个区间的最后一段回文时，左边加上a[l],右边加上a[r]
依然是回文，且次数和dp[l+1][r-1]是一样的，因此再用这种情况更新一下答案
'''

'''
输入约定:
第一行是元素个数n
第二行是n个用空格隔开的数字
'''

'''
测试输入:
8
1 2 3 2 9 8 9 1
测试输出:
2
测试解释:
第一次打掉 2 3 2
     剩下 1 9 8 9 1

剩下的是个回文
所以再打一次即可

'''

inf = 10 ** 18 
#inf表示正无穷
n = int(input())
a = list(map(int, input().split()))

dp = [[inf for i in range(n)] for i in range(n)]
for i in range(n):
	dp[i][i] = 1
for i in range(n - 1):
	if a[i] == a[i + 1]:
		dp[i][i + 1] = 1
	else:
		dp[i][i + 1] = 2

for d in range(2, n):
	for l in range(n):
		r = l + d
		if r >= n: break
		for k in range(l, r):
			dp[l][r] = min(dp[l][r], dp[l][k] + dp[k + 1][r])
		if a[l] == a[r]:
			dp[l][r] = min(dp[l][r], dp[l + 1][r - 1])

'''
for i in range(n):
	s = ''
	for j in range(i, n):
		s += str(dp[i][j]) + ' '
	print(s)
'''

print(dp[0][n - 1])
