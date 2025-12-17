import os
import sys

target_file = r"C:\Users\ascaravilli\AppData\Local\Programs\Python\Python313\Lib\site-packages\mediapipe\tasks\python\core\mediapipe_c_bindings.py"

with open(target_file, 'r') as f:
    content = f.read()

# The block to find
original_block = """  # Register "free()"
  _shared_lib.free.argtypes = [ctypes.c_void_p]
  _shared_lib.free.restype = None"""

# The replacement block
patched_block = """  # Register "free()"
  if hasattr(_shared_lib, 'free'):
      _shared_lib.free.argtypes = [ctypes.c_void_p]
      _shared_lib.free.restype = None
  elif os.name == 'nt':
      _shared_lib.free = ctypes.cdll.msvcrt.free
      _shared_lib.free.argtypes = [ctypes.c_void_p]
      _shared_lib.free.restype = None"""

if original_block in content:
    new_content = content.replace(original_block, patched_block)
    # Create backup
    with open(target_file + ".bak", 'w') as f:
        f.write(content)
    # Write patched file
    with open(target_file, 'w') as f:
        f.write(new_content)
    print("Successfully patched mediapipe_c_bindings.py")
else:
    print("Could not find the original block to patch. It might be already patched or different.")
    # Debug print part of content
    print("Content snippet:")
    print(content[2500:3000])
