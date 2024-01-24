```python
def date2Day(self, year, month, day):
  res = 0
  month_length = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
  for i in range(1971, year):
      res += 366 if self.leap_year(i) else 365
  if self.leap_year(year):
      month_length[2] = 29
  res += sum(month_length[:month]) + day
  return res
```