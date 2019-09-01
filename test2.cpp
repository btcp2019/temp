#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 1e5 + 5;

const int MOD = 1e9 + 7;

int n, m, k, l[N], r[N];

ll dp[N];

int main() {
	cin >> n >> k;
	for (int i = 0; i < n; i ++)
		cin >> l[i] >> r[i], m = max(m, r[i]);
	dp[0] = 1;
	for (int i = 1; i <= m; i ++) {
		dp[i] = dp[i - 1];
		if (i >= k) dp[i] += dp[i - k];
	}
	for (int i = 1; i <= m; i ++) {
		dp[i] += dp[i - 1];
		/*cout << i << ' ' << dp[i] << endl;*/
	}
	for (int i = 0; i < n; i ++)
		cout << /*i << ' ' << l[i] << ' ' << r[i] << ' ' <<*/ dp[r[i]] - dp[l[i] - 1] << '\n';
	return 0;
}
