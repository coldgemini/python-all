import timeit
from time import time

method_time = timeit.timeit(
    "obj.method()",
    """
class SomeClass:
    def method(self):
        pass
obj= SomeClass()
""",
)

start_preprocess_t = time()
