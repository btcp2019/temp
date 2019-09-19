#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e5 + 5;
int n, m;
int vis[N], pre[N], tim;
ll ans;
struct node {
	int a, b, c;
	bool operator < (const node &a) const {
		return c > a.c;
	}
}a[N];
bool dfs(int u) {
	if (vis[u] == tim) return 0;
	vis[u] = tim;
	for (int v : {a[u].a, a[u].b}) 
		if (!pre[v] || dfs(pre[v])) 
			return pre[v] = u, 1;
	return 0;
}
int main() {
	scanf("%d %d", &n, &m);
	for (int i = 1; i <= n; i ++)
		scanf("%d %d %d", &a[i].a, &a[i].b, &a[i].c);
	sort (a + 1, a + n + 1);
	for (tim = 1; tim <= n; tim ++) 
		if (dfs(tim)) ans += a[tim].c;
	printf("%lld\n", ans);
	return 0;
}
