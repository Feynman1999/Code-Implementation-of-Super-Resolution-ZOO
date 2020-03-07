"""
Image Quality Assessment
"""
import importlib

def find_function_using_name(iqa_name):
    """Import the module "iqa/iqa_name.py".

    In the file, the function called iqa_name() will be return. case-insensitive. remove _ in iqa_name
    """
    iqa_filename = "iqa." + iqa_name
    iqalib = importlib.import_module(iqa_filename)
    iqa = None
    target_func_name = iqa_name.replace('_', '')
    for name, func in iqalib.__dict__.items():
        if name.lower() == target_func_name.lower() \
           and hasattr(func, '__call__'):
            iqa = func

    if iqa is None:
        print("In %s.py, there should be a function name that matches %s in lowercase." % (iqa_filename, target_func_name))
        exit(0)

    return iqa