#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 1e5 + 5;

const int MOD = 1e9 + 7;

int n, p, q;

ll fac[N];//阶乘

ll s1, s2;//分子和分母

ll qpow(ll x, ll k, ll Mod = MOD) {//O(logk)求(x^k)%Mod
	ll res = 1;
	for (x %= Mod; k > 0; x = x * x % Mod, k >>= 1)
		if (k & 1) res = res * x % Mod;
	return res;
}

ll calc(ll n, ll m) {//O(logn)求C(n,m)
	return fac[n] * qpow(fac[m], MOD - 2) % MOD * qpow(fac[n - m], MOD - 2) % MOD;
}

int main() {
	cin >> n >> p >> q;
	fac[0] = 1;
	for (int i = 1; i <= n; i ++)
		fac[i] = fac[i - 1] * i % MOD;
	ll tmp;
	for (int i = p; i + q <= n; i ++) {
		tmp = calc(n, i);
		s1 = (s1 + tmp * i % MOD) % MOD;
		s2 = (s2 + tmp) % MOD;
	}
	cout << (s1 * qpow(s2, MOD - 2) % MOD + MOD) % MOD;
	return 0;
}
