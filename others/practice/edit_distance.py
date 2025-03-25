from math import inf
from functools import lru_cache

@lru_cache
def f(i, j, s1, s2):
    if i == len(s1):
        return -2 * len(s2[j:])
    if j == len(s2):
        return -2 * len(s1[i:])
    res = -inf
    if s1[i] == s2[j]:
        res = max(res, f(i + 1, j + 1, s1, s2) + 1)
    else:
        res = max(res, f(i + 1, j + 1, s1, s2) - 1, f(i + 1, j, s1, s2) - 2, f(i, j + 1, s1, s2) - 2)
    return res 

# Test cases
def test_case_1():
    s1 = 'CAAGACG'
    s2 = 'AAGAACG'
    assert f(0, 0, s1, s2) == 2

def test_case_2():
    s1 = 'AACTGA'
    s2 = 'ACTGT'
    assert f(0, 0, s1, s2) == 1

def test_case_3():
    s1 = 'AA'
    s2 = 'A'
    assert f(0, 0, s1, s2) == -1

def test_case_4():
    s1 = 'AA'
    s2 = 'C'
    assert f(0, 0, s1, s2) == -3

def test_case_5():
    s1 = 'AAAC'
    s2 = ''
    assert f(0, 0, s1, s2) == -8

def test_case_6():
    s1 = ''
    s2 = ''
    assert f(0, 0, s1, s2) == 0

if __name__ == '__main__':
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
