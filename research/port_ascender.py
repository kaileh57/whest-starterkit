from __future__ import annotations

import argparse
from pathlib import Path


def expose_masks(src: str) -> str:
    outer_mask = "        except Exception:\n            return fnp.zeros((depth, width), dtype=fnp.float64)"
    outer_expose = (
        "        except Exception as _research_exc:\n"
        "            import traceback as _research_tb\n"
        "            print('ASCENDER_OUTER_FAILURE:', repr(_research_exc), flush=True)\n"
        "            _research_tb.print_exc()\n"
        "            raise"
    )
    if outer_mask not in src:
        raise ValueError("could not find final zero-fallback mask")
    src = src.rsplit(outer_mask, 1)[0] + outer_expose + src.rsplit(outer_mask, 1)[1]

    inner_mask = "                except Exception:\n                    means = None"
    inner_expose = (
        "                except Exception as _k3_exc:\n"
        "                    import traceback as _k3_tb\n"
        "                    print('ASCENDER_K3_FAILURE:', repr(_k3_exc), flush=True)\n"
        "                    _k3_tb.print_exc()\n"
        "                    raise"
    )
    if inner_mask not in src:
        raise ValueError("could not find inner k3 fallback mask")
    return src.replace(inner_mask, inner_expose, 1)


def port_source(src: str, *, expose: bool) -> str:
    # v0.10 makes flopscope arrays intentionally immutable and no longer routes
    # mutation through real NumPy. The public file predates that contract.
    src = src.replace("np.fill_diagonal(", "fnp.fill_diagonal(")

    aug_patch = r'''
# v0.10 compatibility bridge for the embedded pre-v0.10 port_np source.
# Python's augmented assignment rebinds a local name to the value returned by
# __iadd__/etc.; returning the functional operation is therefore equivalent for
# name/attribute targets while preserving FLOP accounting.
try:
    from flopscope._array import FlopscopeArray as _WhestFlopscopeArray
except Exception:
    _WhestFlopscopeArray = None
if _WhestFlopscopeArray is not None:
    _WhestFlopscopeArray.__iadd__ = lambda self, other: fnp.add(self, other)
    _WhestFlopscopeArray.__isub__ = lambda self, other: fnp.subtract(self, other)
    _WhestFlopscopeArray.__imul__ = lambda self, other: fnp.multiply(self, other)
    _WhestFlopscopeArray.__itruediv__ = lambda self, other: fnp.true_divide(self, other)
    _WhestFlopscopeArray.__ifloordiv__ = lambda self, other: fnp.floor_divide(self, other)
    _WhestFlopscopeArray.__ipow__ = lambda self, other: fnp.power(self, other)
    _WhestFlopscopeArray.__imatmul__ = lambda self, other: fnp.matmul(self, other)
'''
    marker = "_backend.enable_flopscope()"
    if marker not in src:
        raise ValueError("could not find enable_flopscope marker")
    src = src.replace(marker, marker + "\n" + aug_patch, 1)
    return expose_masks(src) if expose else src


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expose", action="store_true")
    args = parser.parse_args()
    src = args.input.read_text(encoding="utf-8")
    out = port_source(src, expose=args.expose)
    args.output.write_text(out, encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
