"""Download the OT-1 / OT-2 test data this testbed needs from Google Drive.

Only the files the analysis actually reads are listed.  Videos are the biggest
items by far, so each one is converted to a mono 44.1 kHz WAV after download --
the acoustics code only ever wants the audio.  Pass --keep-video to keep the mp4.

    uv run python testbed/fetch_data.py           # h5 only
    uv run python testbed/fetch_data.py --audio   # h5 + the two spectrogram videos
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# (campaign, filename, drive_id, kind)  kind: "data" | "video"
FILES = [
    # ---- OT-2, "20260701 AEL J1 OTP".  101-108 failed GG ignitions -- skipped.
    ("OT-2", "20260701-109.h5", "1kWzhiYIOMFuDqcsGvPm7-ZfZE2z6GDyM", "data"),
    ("OT-2", "20260701-110.h5", "16Sd5FwFjyG6AwS9QENUNU4he44lqDkPL", "data"),
    ("OT-2", "20260701-111.h5", "1DHk78DOcrsKnPYkd25PEYid8FiBXElFz", "data"),
    # The camera the OT-2 notebook read its spectrogram from.
    ("OT-2", "20260701-111_Canon_700D.mp4", "1aAmcRl08KkGrdxbP4o0ZZ39yrlOpKhXW", "video"),
    # ---- OT-1, R2S 2025.
    ("OT-1", "20250709-001-release.h5", "1D6EBOO9U9QaT_x2HiEw6rchL8ogZjweZ", "data"),
    ("OT-1", "20250709-002-release.h5", "1PLFe19HBrm-eRzQ0jPjKWuk0jGISXc1M", "data"),
    ("OT-1", "20250709-003-release.h5", "1r8qea8Jrp2lsOuoe2OlVfE5HVGf7m7Ec", "data"),
    ("OT-1", "20250709-004-release.h5", "1eAel_AanZfsfcmm6ADYypMPSHUrI3iRe", "data"),
    # The exact video the OT-1 notebook read its RPM trace from (exp #2).
    ("OT-1", "20250709-002_2160p60.mp4", "1qOPbU16CR7ZkKe7h2wyH9Ebxngk7NmgA", "video"),
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def drive_download(file_id: str, dest: Path) -> None:
    """Fetch a Drive file that is shared by link, handling the virus-scan gate."""
    cookies = dest.with_suffix(dest.suffix + ".cookies")
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        subprocess.run(
            ["curl", "-sS", "-L", "-A", UA, "-c", str(cookies), "-b", str(cookies),
             "--retry", "3", "--retry-delay", "2", url, "-o", str(dest)],
            check=True,
        )
    finally:
        cookies.unlink(missing_ok=True)

    # A gate page is HTML and small; a real payload is neither.
    head = dest.open("rb").read(512)
    if head.lstrip()[:1] == b"<":
        dest.unlink()
        raise RuntimeError(f"Drive returned an HTML page for {file_id}, not the file")


def extract_wav(mp4: Path) -> Path:
    wav = mp4.with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "44100", str(wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true", help="also fetch the two videos")
    ap.add_argument("--keep-video", action="store_true", help="keep the mp4 after WAV extraction")
    args = ap.parse_args()

    for campaign, name, file_id, kind in FILES:
        if kind == "video" and not args.audio:
            continue
        out = DATA / campaign / name
        out.parent.mkdir(parents=True, exist_ok=True)

        if kind == "video" and out.with_suffix(".wav").exists():
            print(f"  have  {campaign}/{out.with_suffix('.wav').name}")
            continue
        if out.exists() and out.stat().st_size > 0:
            print(f"  have  {campaign}/{name}")
        else:
            print(f"  get   {campaign}/{name} ...", flush=True)
            drive_download(file_id, out)
            print(f"        {out.stat().st_size / 1e6:.1f} MB")

        if kind == "video":
            wav = extract_wav(out)
            print(f"        -> {wav.name}  ({wav.stat().st_size / 1e6:.1f} MB)")
            if not args.keep_video:
                out.unlink()
                print("        mp4 removed (pass --keep-video to keep it)")

    print(f"\ndata root: {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
