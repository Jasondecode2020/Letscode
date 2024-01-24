```python
def leap_year(self, year):
  return (year % 400 == 0) or (year % 100 != 0 and year % 4 == 0)
```