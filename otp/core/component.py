from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import dataclass_transform


def q(unit="", desc="", fmt=".3g", *, alt=None,
      default=MISSING, default_factory=MISSING, init=True):
    kw = {}
    if default is not MISSING:
        kw["default"] = default
    if default_factory is not MISSING:
        kw["default_factory"] = default_factory
    return field(metadata={"unit": unit, "desc": desc, "fmt": fmt, "alt": alt},
                 init=init, **kw)

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

def _rows(obj, indent=0):
    out = []
    for f in fields(obj):
        v = getattr(obj, f.name, None)
        name = "  " * indent + f.name + ("" if f.init else "*")
        if is_dataclass(v):
            out.append((name, "", "", "", "", type(v).__name__))
            out += _rows(v, indent + 1)
        else:
            m = f.metadata
            alt = m.get("alt")
            if alt and isinstance(v, (int, float)) and not isinstance(v, bool):
                alt_u, conv = alt
                alt_v = format(conv(v) if callable(conv) else v * conv, m.get("fmt", ".3g"))
            else:
                alt_v = alt_u = ""
            out.append((name, _fmt(v, m), m.get("unit", ""), alt_v, alt_u, m.get("desc", "")))
    return out

def _show(self):
    rows = _rows(self)
    wn = max(len(r[0]) for r in rows) + 2
    wv = max(len(r[1]) for r in rows) + 2
    wu = max(len(r[2]) for r in rows) + 2
    if any(r[3] for r in rows):
        wa = max(len(r[3]) for r in rows)
        wau = max(len(r[4]) for r in rows)
        body = "\n".join(f"  {n:<{wn}} {v:>{wv}} {u:<{wu}} {a:>{wa}} {au:<{wau}}  {d}"
                         for n, v, u, a, au, d in rows)
    else:
        body = "\n".join(f"  {n:<{wn}} {v:>{wv}} {u:<{wu}}  {d}"
                         for n, v, u, _, _, d in rows)
    return f"{type(self).__name__}\n{body}"

@dataclass_transform(field_specifiers=(field, q))
def component(cls):
    cls = dataclass(cls)
    cls.__str__ = cls.__repr__ = _show
    return cls