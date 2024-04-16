#cc -c -o gpssim.o gpssim.c -fPIC | gcc -shared -o gpssim.so gpssim.o
#hackrf_transfer -t gpssim.bin -f 1575420000 -s 2600000 -a 1 -x 0
import ctypes
import struct

# A. Create library
C_library = ctypes.CDLL("./gpssim.so")

# B. Specify function signatures
initialize = C_library.ephem_initial
initialize.argtypes = [ctypes.c_char_p]

# C. Invoke function
filename = "brdc1070.24n".encode('ascii')
ret = initialize(filename) 

# B. Specify function signatures
get_stream = C_library.get_signal
get_stream.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double]

# C. Invoke function
y = get_stream(37.496248,126.968514,10.0,0.0) # hyncungwon
print(y)

