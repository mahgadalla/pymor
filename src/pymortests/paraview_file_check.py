#!/usr/bin/env python2

if __name__ == '__main__':
    import sys

    try:
        import paraview.simple as ps
    except ImportError:
        sys.exit(77)

    reader = ps.OpenDataFile(sys.argv[1])
    ps.Show()
