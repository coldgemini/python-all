# print(**a_dict)

def testf(haha, hoho):
    print(haha)
    print(hoho)


if __name__ == '__main__':
    a_dict = {"haha": 3, "hoho": 4}
    testf(**a_dict)
