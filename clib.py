#cc -c -o gpssim.o gpssim.c -fPIC
#gcc -shared -o gpssim.so gpssim.o

import ctypes

# A. Create library
C_library = ctypes.CDLL("./gpssim.so")

# B. Specify function signatures
hello_fxn = C_library.get_signal
hello_fxn.argtypes = [ctypes.c_float,ctypes.c_float,ctypes.c_float]

# C. Invoke function

y = hello_fxn(35.681298,139.766247,10)
print(y)

