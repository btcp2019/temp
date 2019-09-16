import sys
import math

def sqr(x):
	return x * x

def dist(a, b):
	return sqr(a[0] - b[0]) + sqr(a[1] - b[1])

f = [0 for i in range(233)]

def fa(x):
	if x == f[x]:
		return x
	f[x] = fa(f[x])
	return f[x]

def un(x, y):
	x, y = fa(x), fa(y)
	if x == y:
		return
	f[x] = y

for s in sys.stdin.readlines():
	s = s.split()
	n, r, d = int(s[0]), float(s[1]), float(s[2])
	#print(n)
	p = [[float(s[i * 2 + 3]), float(s[i * 2 + 4])] for i in range(n)]
	#print(*p)
	for i in range(1, n + 2 + 1):
		f[i] = i
	for i in range(1, n):
		for j in range(i + 1, n + 1):
			if dist(p[i - 1], p[j - 1]) <= sqr(r + r):
				un(i, j)
	for i in range(1, n + 1):
		if p[i - 1][1] + r >= 100:
			un(i, n + 2)
		if p[i - 1][1] - r <= 0:
			un(i, n + 1)
	if fa(n + 1) == fa(n + 2):
		print('N')
	elif d == 0:
		print('Y')
	else:
		dis = [[0 for i in range(n + 3)] for i in range(n + 3)]
		for i in range(1, n + 1):
			for j in range(1, n + 1):
				#print(i, j, math.sqrt(dist(p[i - 1], p[j - 1])))
				if dist(p[i - 1], p[j - 1]) > sqr(r + r):
					dis[i][j] = -int((r + r - math.sqrt(dist(p[i - 1], p[j - 1]))) // (d * 2))
		for i in range(1, n + 1):
			if p[i - 1][1] + r < 100:
				dis[n + 2][i] = dis[i][n + 2] = -int((p[i - 1][1] + r - 100) // (d * 2))
			if p[i - 1][1] - r > 0:
				dis[n + 1][i] = dis[i][n + 1] = -int((r - p[i - 1][1]) // (d * 2))
		dis[n + 1][n + 2] = dis[n + 2][n + 1] = -int(-100 // (d * 2))
		#print(*dis)
		for k in range(1, n + 3):
			for i in range(1, n + 3):
				for j in range(1, n + 3):
					dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
		#print(*dis)
		print(dis[n + 1][n + 2])