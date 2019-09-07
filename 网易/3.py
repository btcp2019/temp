for i in range(int(input())):
	k, m = map(int, input().split())
	if m == 0:
		print(1 + (30 - 1) // (k + 1))
	else:
		d = list(map(int, input().split()))
		print((d[0] - 1) // (k + 1) + (30 - d[-1]) // (k + 1) 
			+ sum(max(0, d[i] - d[i - 1] - 1 - k) // (k + 1) for i in range(1, m))
			+ m)