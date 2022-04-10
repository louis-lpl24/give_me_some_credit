def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True

        else:
            return False

    except NameError as _:
        return False
