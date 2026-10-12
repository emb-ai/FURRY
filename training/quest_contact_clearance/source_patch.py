"""Checked edits to pinned source methods, retaining inspectable source."""
import inspect
import linecache
import textwrap


def edited(fn, replacements, label):
    source = textwrap.dedent(inspect.getsource(fn))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Pinned source changed: %s: %r' % (label, old))
        source = source.replace(old, new)
    filename = '<quest-contact-clearance:%s>' % label
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(fn.__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[fn.__name__]
