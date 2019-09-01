#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 1e6 + 6;

int n, m;

char s[N], t[N];

int nxt[N];

int ok[N];

void kmp(int n, char *a) {
	//长度为m的b中找a,下标从0开始,得到的是匹配成功的末尾位置
	static int i, j;
	for (nxt[0] = j = -1, i = 1; i < n; nxt[i ++] = j) {
		while (~j && a[j + 1] != a[i]) j = nxt[j];
		if (a[j + 1] == a[i]) j ++;
	}
}

int dfs(int x) {
	if (ok[x] != -1) return ok[x];
	if (x * 2 + 1 >= n) {
		for (int i = 0, j = x + 1; j < n; i ++, j ++)
			if (s[i] != s[j])
				return ok[x] = 0;
		return ok[x] = 1;
	}
	return ok[x] = nxt[x * 2 + 1] == x && dfs(x * 2 + 1);
}

bool judge(char *t, int n) {
	for (int i = 0; i < n; i ++)
		if (t[i] != s[i])
			return 0;
	return 1;
}

int main() {
	ios::sync_with_stdio(false);
	cin >> n >> s;
	kmp(n, s);
	for (int i = 0; i < n; i ++)
		ok[i] = -1;
	for (int i = 0; i < n; i ++)
		dfs(i);
	cin >> m; int ans = 0;
	for (int len, i = 0; i < m; i ++) {
		cin >> t; len = strlen(t);
		if (!ok[len - 1] || !judge(t, len));
		else ans ++; 
	}
	cout << ans;
	return 0;
}
