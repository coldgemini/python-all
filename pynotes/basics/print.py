print("{0} {1}".format(1, 2))
'{:10}'.format('tests')
'{:>10}'.format('tests')
'{:_<10}'.format('tests')
'{:^10}'.format('tests')

'{:d}'.format(42)
'{:4d}'.format(42)
'{:04d}'.format(42)
'{:f}'.format(3.141592653589793)
'{:06.2f}'.format(3.141592653589793)


val = 12.3

print(f'{val:.2f}')
print(f'{val:.5f}')

for x in range(1, 11):
    print(f'{x:02} {x*x:3} {x*x*x:4}')