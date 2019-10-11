from contextlib import contextmanager
import time
from datetime import datetime


@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


@contextmanager
def long_run(name):
    try:
        print("entering context")
        yield
    except KeyboardInterrupt:
        # print('Interrupted in model')
        print('Interrupted in model')
        raise KeyboardInterrupt
    except Exception as e:
        # print('Exception in model')
        print('Exception in model')
        raise e
    finally:
        print(("{}: Done " + name).format(datetime.now().strftime("%m/%d %H:%M:%S")))


def main():
    with long_run("main"):
        for i in range(100):
            print("in long run")
            time.sleep(1)


if __name__ == '__main__':
    main()
