"""Inspect .pyc files to recover structure for reconstruction."""
import dis, marshal, types

def inspect_pyc(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        flags = int.from_bytes(f.read(4), 'little')
        if flags & 0x1:
            f.read(8)
        else:
            f.read(8)
        code = marshal.load(f)
    
    print(f'=== {path} ===')
    print(f'Name: {code.co_name}')
    print(f'Names: {code.co_names}')
    print()
    
    def walk_code(co, indent=0):
        prefix = '  ' * indent
        print(f'{prefix}Code: {co.co_name} (args: {co.co_varnames[:co.co_argcount]})')
        # Show string constants
        for c in co.co_consts:
            if isinstance(c, str) and len(c) > 2:
                print(f'{prefix}  str: {c!r}')
        # Show names used
        print(f'{prefix}  names: {co.co_names}')
        print(f'{prefix}  varnames: {co.co_varnames}')
        # Recurse into nested code objects
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                walk_code(c, indent + 1)
    
    walk_code(code)
    print()

inspect_pyc('simplex_splat/__pycache__/metrics.cpython-313.pyc')
inspect_pyc('simplex_splat/__pycache__/monitor.cpython-313.pyc')
inspect_pyc('simplex_splat/__pycache__/__init__.cpython-313.pyc')
