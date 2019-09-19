#include <bits/stdc++.h>
using namespace std;
const int N = 555;
int n, m, k;
vector<int>e[N];
int isLoop, cnt;
int vis[N];
void dfs(int u) {
	vis[u]=1, cnt++;
	for(int v:e[u]){
		if (vis[v]) isLoop=1;
		else dfs(v);
	}
}
int main() {
	scanf("%d %d", &n, &m), k = n;
	for (int u, v, i = 1; i <= m; i ++) {
		scanf("%d %d", &u, &v);
		e[u].push_back(v);
		e[v].push_back(u);
	}
	for (int i = 1; i <= n; i ++) {
		if (e[i].size() == 0 || vis[i]) continue;
		isLoop = 0, cnt = 0, dfs(i);
		k -= (cnt&1)&(isLoop);
	}
	k-=k&1;
	cout<<k;
	return 0;
}
