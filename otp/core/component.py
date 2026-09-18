from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import dataclass_transform

# ---- sections -------------------------------------------------------------

_SEP = "/"            # path separator for group names
_GROUPS = []          # active group path while a class body is executing

def _path(names):
    out = []
    for n in names:
        if n is None:
            continue
        if isinstance(n, str):
            out += [p.strip() for p in n.split(_SEP) if p.strip()]
        else:
            out += list(n)
    return tuple(out)

@contextmanager
def group(*names):
    """Group every q() declared in the body. Nest for subgroups."""
    p = _path(names)
    _GROUPS.extend(p)
    try:
        yield
    finally:
        if p:
            del _GROUPS[-len(p):]

# ---- field spec -----------------------------------------------------------

def q(unit="", desc="", fmt=".3g", *, alt=None, group=None,
      default=MISSING, default_factory=MISSING, init=True):
    kw = {}
    if default is not MISSING:
        kw["default"] = default
    if default_factory is not MISSING:
        kw["default_factory"] = default_factory
    return field(metadata={"unit": unit, "desc": desc, "fmt": fmt, "alt": alt,
                           "group": tuple(_GROUPS) + _path([group])},
                 init=init, **kw)

# ---- printing -------------------------------------------------------------

GROUP_PREFIX = "----- "
GROUP_SUFFIX = " -----"

def _fmt(v, meta):
    if v is None:
        return "..."
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return format(v, meta.get("fmt", ".3g"))
    if type(v).__str__ is object.__str__ and type(v).__repr__ is object.__repr__:
        label = getattr(v, "name", None)
        if label is None or callable(label):
            return f"<{type(v).__name__}>"
        return f"<{type(v).__name__} {label}>"
    return str(v)

def _node():
    return {"items": [], "subs": {}}

def _tree(obj):
    """Fields in declaration order, folded into a group tree.
    A group's block sits where its first member was declared."""
    root = _node()
    for f in fields(obj):
        node = root
        for key in f.metadata.get("group") or ():
            if key not in node["subs"]:
                node["subs"][key] = _node()
                node["items"].append(("group", key))
            node = node["subs"][key]
        node["items"].append(("field", f))
    return root

def _emit(node, obj, indent, out):
    pad = "  " * indent
    for kind, item in node["items"]:
        if kind == "group":
            out.append(("group", pad + GROUP_PREFIX + item + GROUP_SUFFIX, "", "", "", "", "", indent))
            _emit(node["subs"][item], obj, indent + 1, out)
            continue
        f = item
        v = getattr(obj, f.name, None)
        name = pad + f.name + ("" if f.init else "*")
        if is_dataclass(v):
            out.append(("row", name, "", "", "", "", type(v).__name__, indent))
            out.extend(_rows(v, indent + 1))
        else:
            m = f.metadata
            alt = m.get("alt")
            if alt and isinstance(v, (int, float)) and not isinstance(v, bool):
                alt_u, conv = alt
                alt_v = format(conv(v) if callable(conv) else v * conv, m.get("fmt", ".3g"))
            else:
                alt_v = alt_u = ""
            out.append(("row", name, _fmt(v, m), m.get("unit", ""),
                        alt_v, alt_u, m.get("desc", ""), indent))

def _rows(obj, indent=0):
    out = []
    _emit(_tree(obj), obj, indent, out)
    return out

# ---- whole-object and single-section rendering ----------------------------

class _Text(str):
    __repr__ = str.__str__

def _format(title, rows):
    data = [r for r in rows if r[0] == "row"] or rows
    wn = max(len(r[1]) for r in data) + 2
    wv = max(len(r[2]) for r in data) + 2
    wu = max(len(r[3]) for r in data) + 2
    use_alt = any(r[4] for r in data)
    if use_alt:
        wa = max(len(r[4]) for r in data)
        wau = max(len(r[5]) for r in data)

    lines = []
    for i, (kind, n, v, u, a, au, d, lvl) in enumerate(rows):
        # prev = rows[i - 1] if i else None
        # if prev and prev[0] != "group" and (kind == "group" or lvl < prev[7]):
        #     lines.append("")
        if kind == "group":
            lines.append(f"  {n}")
        elif use_alt:
            lines.append(f"  {n:<{wn}} {v:>{wv}} {u:<{wu}} {a:>{wa}} {au:<{wau}}  {d}")
        else:
            lines.append(f"  {n:<{wn}} {v:>{wv}} {u:<{wu}}  {d}")
    return _Text(f"{title}\n" + "\n".join(lines))

def _show(self):
    return _format(type(self).__name__, _rows(self))

def _targets(obj, node=None, path=(), out=None):
    """Every addressable block: each group, and each subcomponent field."""
    if out is None:
        out, node = [], _tree(obj)
    for kind, item in node["items"]:
        if kind == "group":
            sub = node["subs"][item]
            out.append((path + (item,), "group", sub, obj))
            _targets(obj, sub, path + (item,), out)
        else:
            v = getattr(obj, item.name, None)
            if is_dataclass(v):
                out.append((path + (item.name,), "obj", v, None))
                _targets(v, _tree(v), path + (item.name,), out)
    return out

def _sections(self):
    """Every name you can pass to .section(), as 'a/b' paths."""
    return [_SEP.join(p) for p, *_ in _targets(self)]

def _section(self, *names):
    want = _path(names)
    if not want:
        raise ValueError(f"nothing requested; available: {_sections(self)}")
    hits = [t for t in _targets(self)
            if t[0][-len(want):] == want]
    if not hits:
        raise ValueError(f"no section {_SEP.join(want)!r}; available: {_sections(self)}")

    blocks = []
    for path, kind, payload, owner in hits:
        rows = []
        if kind == "group":
            _emit(payload, owner, 0, rows)
        else:
            rows = _rows(payload, 0)
        title = f"{type(self).__name__} {_SEP} " + f" {_SEP} ".join(path)
        blocks.append(_format(title, rows))
    return _Text("\n\n".join(blocks))

@dataclass_transform(field_specifiers=(field, q))
def component(cls):
    cls = dataclass(cls)
    cls.__str__ = cls.__repr__ = _show
    cls.section = _section
    cls.sections = _sections
    return cls