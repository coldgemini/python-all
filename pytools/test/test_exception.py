class NoClineError(Exception):
    """raised when multiple centerlines generated"""
    pass


try:
    raise NoClineError("No Cline !!")
except Exception as ex:
    print(str(ex))
