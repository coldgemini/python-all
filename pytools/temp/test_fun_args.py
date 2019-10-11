def foo(x, *args, **kwargs):
    print(x)
    print(type(args))
    print(args)
    print(kwargs)


def main():
    # foo(1, 2, 3, 4, y=1, a=2, b=3, c=4)
    l1 = [2, 3, 4]
    # foo(1, *l1, y=1, a=2, b=3, c=4)
    print(type(*l1))


if __name__ == '__main__':
    main()
