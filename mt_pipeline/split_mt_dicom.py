import argparse
import copy
import os

import pydicom
from pydicom.uid import generate_uid


def split_mt_enhanced(in_path, out_off, out_on, off_first_nframes=None):
    ds = pydicom.dcmread(in_path)
    nframes = int(ds.NumberOfFrames)

    # Try TemporalPositionIndex; fallback to first half vs second half.
    tpos = []
    if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        for fg in ds.PerFrameFunctionalGroupsSequence:
            try:
                tpos.append(int(fg.FrameContentSequence[0].TemporalPositionIndex))
            except Exception:
                tpos.append(None)
    uniq = sorted(x for x in set(tpos) if x is not None)
    if len(uniq) == 2:
        idx_off = [i for i, p in enumerate(tpos) if p == uniq[0]]
        idx_on = [i for i, p in enumerate(tpos) if p == uniq[1]]
    else:
        if off_first_nframes is None:
            off_first_nframes = nframes // 2
        idx_off = list(range(off_first_nframes))
        idx_on = list(range(off_first_nframes, nframes))

    rows, cols = ds.Rows, ds.Columns
    spp = getattr(ds, "SamplesPerPixel", 1)
    bits = ds.BitsAllocated
    bpf = rows * cols * spp * bits // 8
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        raise RuntimeError("Pixel data is compressed; decompress before splitting.")

    def subset(ds_src, idxs, series_suffix):
        new = copy.deepcopy(ds_src)
        new.NumberOfFrames = len(idxs)

        pixels = ds_src.PixelData
        chunks = []
        for i in idxs:
            start = i * bpf
            chunks.append(pixels[start : start + bpf])
        new.PixelData = b"".join(chunks)

        if hasattr(new, "PerFrameFunctionalGroupsSequence"):
            new.PerFrameFunctionalGroupsSequence = [
                ds_src.PerFrameFunctionalGroupsSequence[i] for i in idxs
            ]
            for k, fg in enumerate(new.PerFrameFunctionalGroupsSequence, start=1):
                if "FrameContentSequence" in fg:
                    fcs = fg.FrameContentSequence[0]
                    if hasattr(fcs, "TemporalPositionIndex"):
                        fcs.TemporalPositionIndex = 1
                    if hasattr(fcs, "InStackPositionNumber"):
                        fcs.InStackPositionNumber = k

        if hasattr(new, "NumberOfTemporalPositions"):
            new.NumberOfTemporalPositions = 1

        new.SOPInstanceUID = generate_uid()
        new.SeriesInstanceUID = generate_uid()
        if hasattr(new, "SeriesDescription"):
            base = getattr(ds_src, "SeriesDescription", "")
            new.SeriesDescription = f"{base}_{series_suffix}" if base else series_suffix
        new.InstanceNumber = 1
        return new

    ds_off = subset(ds, idx_off, "MT_off")
    ds_on = subset(ds, idx_on, "MT_on")

    # Ensure output folders exist (e.g., .../MToff, .../MTon)
    for path in (out_off, out_on):
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)

    ds_off.save_as(out_off)
    ds_on.save_as(out_on)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Split an MT enhanced multi-frame DICOM into off/on volumes."
    )
    parser.add_argument(
        "input",
        help="Input enhanced MR DICOM path, or a directory containing a single DICOM file",
    )
    parser.add_argument("out_off", help="Output DICOM for MT-off frames")
    parser.add_argument("out_on", help="Output DICOM for MT-on frames")
    parser.add_argument(
        "--off-first-nframes",
        type=int,
        default=None,
        help="Frames to assign to MT-off when no temporal index exists (default: half).",
    )
    args = parser.parse_args(argv)
    in_path = resolve_input_path(args.input)
    split_mt_enhanced(
        in_path, args.out_off, args.out_on, off_first_nframes=args.off_first_nframes
    )


def resolve_input_path(path):
    """Accept a file path or a directory with exactly one DICOM file."""
    p = os.path.expanduser(path)
    if os.path.isdir(p):
        files = [
            os.path.join(p, f)
            for f in os.listdir(p)
            if os.path.isfile(os.path.join(p, f))
        ]
        def is_generated(name):
            b = os.path.basename(name).lower()
            return b.startswith("mtoff") or b.startswith("meton")

        filtered = [f for f in files if not is_generated(f)]
        dcm_like = [f for f in filtered if f.lower().endswith(".dcm")]
        if len(dcm_like) == 1:
            return dcm_like[0]
        if len(filtered) == 1:
            return filtered[0]
        if len(dcm_like) == 1:
            return dcm_like[0]
        if len(files) == 1:
            return files[0]
        raise ValueError(
            f"Directory {p} must contain exactly one DICOM file; found {len(files)} files."
        )
    return p


if __name__ == "__main__":
    main()
