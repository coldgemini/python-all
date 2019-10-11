from unittest import mock


# import myobj

# class MyObj():
#     def __init__(self, repo):
#         repo.connect()

class MyObj():
    def __init__(self, repo):
        self._repo = repo
        repo.connect()

    def setup(self):
        print("return value: {}".format(self._repo.setup(cache=True)))

    def custom1(self):
        self._repo.custom1(cache=True)


def test_instantiation():
    external_obj = mock.Mock()
    MyObj(external_obj)
    external_obj.connect.assert_called_with()


def test_setup():
    external_obj = mock.Mock()
    external_obj.setup.return_value = 3
    obj = MyObj(external_obj)
    obj.setup()
    # external_obj.setup.assert_called_with(cache=True, max_connections=256)
    external_obj.setup.assert_called_with(cache=True)


def test_custom1():
    external_obj = mock.Mock()
    obj = MyObj(external_obj)
    obj.custom1()
    external_obj.setup.assert_not_called()
    external_obj.custom1.assert_called_with(cache=True)


import os
from unittest.mock import patch


class FileInfo:
    def __init__(self, path):
        self.original_path = path
        self.filename = os.path.basename(path)

    def get_info(self):
        print(os.path.abspath(self.filename))
        return self.filename, self.original_path, os.path.abspath(self.filename)


def test_init():
    filename = 'somefile.ext'
    fi = FileInfo(filename)
    assert fi.filename == filename


def test_init2():
    filename = 'somefile.ext'
    relative_path = '../{}'.format(filename)
    fi = FileInfo(relative_path)
    assert fi.filename == filename


def test_get_info():
    filename = 'somefile.ext'
    original_path = '../{}'.format(filename)

    with patch('os.path.abspath') as abspath_mock:
        test_abspath = 'some/abs/path'
        abspath_mock.return_value = test_abspath
        fi = FileInfo(original_path)
        assert fi.get_info() == (filename, original_path, test_abspath)
