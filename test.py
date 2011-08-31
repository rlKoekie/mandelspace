def foo():
    import re
    return re.__dict__

d=foo()
print d
print d["TEMPLATE"]
