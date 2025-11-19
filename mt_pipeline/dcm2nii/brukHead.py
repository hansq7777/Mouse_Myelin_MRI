import sys
import json
from pathlib import Path
import pydicom
from pybruker import jcamp
import numpy as np


def main():
    # Strip off '.dcm'
    fname = sys.argv[1]
    fname_noExt = fname[0:-4]

    method = None
    visu_pars = None

    # read dicom header
    H = pydicom.read_file(fname, stop_before_pixels=True)

    # try to use embedded Bruker private tags first
    tag_method = (0x0177, 0x1100)
    tag_visu = (0x0177, 0x1101)
    if tag_method in H and tag_visu in H:
        method = jcamp.jcamp_parse(
            H[tag_method].value.decode('utf-8').splitlines()
        )
        visu_pars = jcamp.jcamp_parse(
            H[tag_visu].value.decode('utf-8').splitlines()
        )
    else:
        # fallback: look for method and visu_pars files on disk
        dcm_path = Path(fname).resolve()
        search_dirs = [dcm_path.parent] + list(dcm_path.parents)
        method_path = None
        visu_path = None
        for p in search_dirs:
            candidate = p / "method"
            if candidate.is_file():
                method_path = candidate
                break
        for p in search_dirs:
            candidate = p / "visu_pars"
            if candidate.is_file():
                visu_path = candidate
                break
        if method_path and visu_path:
            with open(method_path, "r", encoding="utf-8", errors="ignore") as f:
                method = jcamp.jcamp_parse(f.read().splitlines())
            with open(visu_path, "r", encoding="utf-8", errors="ignore") as f:
                visu_pars = jcamp.jcamp_parse(f.read().splitlines())
        else:
            msg = (
                f"Bruker private tags missing in {fname} and no method/visu_pars files found; "
                "skipping JSON/bval/bvec extraction."
            )
            print(msg)
            return 0

    with open(fname_noExt + '_method.json', 'w') as fp:
        json.dump(method, fp, indent=4)

    with open(fname_noExt + '_visu_pars.json', 'w') as fp:
        json.dump(visu_pars, fp, indent=4)

    # For MT-only workflows, stop after writing method/visu_pars JSON.
    return 0


if __name__ == "__main__":
    sys.exit(main())
