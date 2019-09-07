for i in range(int(input())):
	n = int(input())
	a = [list(map(int, input().split())) for i in range(n)]
	s = [0 for i in range(n)]
	rd = [0 for i in range(n)]
	for val, lson, rson in a:
		if lson != -1:
			rd[lson] += 1
		if rson != -1:
			rd[rson] += 1
	root = rd.index(0)

	q, siz, max_dep = [[root, 0]], 1, 0
	while siz != 0:
		u, dep = q[-1]
		s[dep] += a[u][0]
		max_dep = max(max_dep, dep)

		q.pop()
		siz -= 1
		
		if a[u][1] != -1:
			q.append([a[u][1], dep + 1])
			siz += 1
		if a[u][2] != -1:
			q.append([a[u][2], dep + 1])
			siz += 1


	s = [s[i] - s[i - 1] for i in range(1, max_dep)]

	print(['YES', 'NO'][any(i <= 0 for i in s)])
