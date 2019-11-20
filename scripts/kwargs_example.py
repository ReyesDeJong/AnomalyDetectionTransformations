"""
def example2(arg_1, kw_1="shark", kw_2="blobfish", **kwargs):
  print(arg_1)
  print(kw_1, kw_2)
  print(kwargs)
  print(kwargs['name'])
example2(1, name=1, **{'kw_1': 1, 'kw_2': 'b'})
1
1 b
{'name': 1}
1

example2(1, **{'name': 1, 'kw_2': 'b'})
1
shark b
{'name': 1}
1
"""