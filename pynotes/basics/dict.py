def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield subkey, subvalue
            else:
                yield key, value

    return dict(items())


def main():
    adict = {'haha': 3, 'hoho': {'hehe': 5, 'hghg': 0}}
    bditt = flatten_dict(adict)
    print(bditt)


if __name__ == '__main__':
    main()

dict.get(key, default = None)