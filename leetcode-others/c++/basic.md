### 1 vector and string

```c++
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // Write C++ code here
    vector<int> v;
    v.push_back(3);
    v.push_back(5);
    v.push_back(2);
    cout << v[0] << "\n";
    cout << v[1] << "\n";
    cout << v[2] << "\n";
    for (int i = 0; i < v.size(); i ++) {
        cout << v[i] << "\n";
    }
    for (auto x: v) {
        cout << x << "\n";
    }
    cout << v.back() << "\n";
    v.pop_back();
    cout << v.back() << "\n";
    vector<int> v1 = {1, 2, 3};
    vector<int> v2(10);
    cout << v2[0] << " ";
    vector<int> v3(10, 1);
    cout << v3[0] << "\n";
    
    // string
    string a = "abc";
    string b = a + a;
    cout << b << "\n";
    
    b[1] = 'z';
    cout << b << "\n";
    cout << b.find('a') << "\n";
    cout << b.substr(1, 2) << "\n";
    return 0;
}
```

### 2 set

```c++
#include <iostream>
#include <unordered_set>
#include <set>
using namespace std;

int main() {
    // Write C++ code here
    set<int> s;
    s.insert(3);
    s.insert(2);
    s.insert(5);
    cout << s.count(3) << "\n"; // 1
    cout << s.count(4) << "\n"; // 0
    s.erase(3);
    s.insert(4);
    cout << s.count(3) << "\n"; // 0
    cout << s.count(4) << "\n"; // 1
    
    set<int> s1 = {2,5,6,8};
    cout << s1.size() << "\n"; // 4
    for (auto x : s1) {
    cout << x << "\n";
    }
    
    set<int> s2;
    s2.insert(5);
    s2.insert(5);
    s2.insert(5);
    cout << s2.count(5) << "\n"; // 1
    
    multiset<int> s3;
    s3.insert(5);
    s3.insert(5);
    s3.insert(5);
    cout << s3.count(5) << "\n"; // 3

    s3.erase(5);
    cout << s3.count(5) << "\n"; // 0

    multiset<int> s4;
    s4.insert(5);
    s4.insert(5);
    s4.erase(s4.find(5));
    cout << s4.count(5) << "\n"; // 2

    return 0;
}
```