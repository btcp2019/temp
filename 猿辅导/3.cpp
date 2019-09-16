#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

typedef pair<int, int> piir;

const int N = 3e5 + 5;
const int M = 1e4 + 5;

struct bit {
	int lowbit(int x) {
	    return x&(-x);
	}
	 
	void update(ll c[],int x,ll num) {
	    for(int i = x;i < N;i += lowbit(i)) {
	        c[i] += num;
	    }
	}
	 
	ll query(ll c[],ll x) {
	    ll sum = 0;
	    for(int i = x;i >= 1;i -= lowbit(i)) {
	        sum += c[i];
	    }
	    return sum;
	}
	 
	ll c[N];
	ll c2[N];
	ll sum[N];
	
	void add(ll x, ll y, ll num = 1) {
		update(c,x,num);
        update(c,y+1LL,-num);
        update(c2,x,(x-1LL)*num);
        update(c2,y+1LL,-y*num);
	}

	ll ask(ll x, ll y) {
		return (y*query(c,y) - query(c2,y) ) - ( (x-1LL) *query(c,x-1LL) -query(c2,x - 1LL) );
	}
}t;

struct c1ass {
	ll k, stu, tea;
	string name;

	bool operator < (const c1ass &a) const {
		return stu * a.tea > tea * a.stu;//stu/tea > a.stu/a.tea
	}
}a[M];

struct statu{
	char s[5];
	int id, tim;

	bool operator < (const statu &a) {
		if (tim != a.tim) return tim < a.tim;
		return s[0] < a.s[0];//IN < OUT
	}
}b[N];

map <int, int> p;
map <int, int> io;

vector <piir> tea[M];
vector <piir> stu[M]; 

int n, m;

int main() {
	ios::sync_with_stdio(false);
	cin >> n >> m;
	for (int x, i = 1; i <= m; i ++) {
		cin >> a[i].k >> x;
		p[x] = -i;
		cin >> a[i].name;
		for (int j = 1; j <= a[i].k; j ++) {
			cin >> x;
			p[x] = i;
		}
	}
	for (int i = 0; i < n; i ++) 
		cin >> b[i].s >> b[i].id >> b[i].tim;
	sort (b, b + n);
	for (int id, i = 0; i < n; i ++) {
		id = p[b[i].id];
		if (b[i].s[0] == 'I') io[abs(id)] = b[i].tim;
		else {
			if (id < 0) {
				tea[abs(id)].push_back(piir(io[abs(id)], b[i].tim));
			}
			else {
				stu[id].push_back(piir(io[id], b[i].tim));
			}
		}
	}
	for (int i = 1; i <= m; i ++) {
		for (piir j : tea[i]) {
			if (j.first == j.second) continue;
			t.add(j.first, j.second - 1);
			a[i].tea += j.second - j.first;
		}
		for (piir j : stu[i]) {
			if (j.first == j.second) continue;
			a[i].stu += t.ask(j.first, j.second - 1);
		}
		for (piir j : tea[i]) {
			if (j.first == j.second) continue;
			t.add(j.first, j.second - 1);
			a[i].tea += j.second - j.first;
		}
		a[i].tea *= a[i].k;
	}
	sort (a + 1, a + m + 1);
	for (int i = 1; i <= m; i ++)
		cout << a[i].name << '\n';
	return 0;
}