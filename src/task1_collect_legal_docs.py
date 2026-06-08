"""
Task 1 - Collect legal documents about drugs and controlled substances.

README requirement:
    - Download at least 3 legal documents as PDF/DOC/DOCX.
    - Save original files to data/landing/legal/.
    - Use clear filenames, e.g. luat-phong-chong-ma-tuy-2021.pdf.

Run:
    python src/task1_collect_legal_docs.py
"""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
MIN_FILE_SIZE_BYTES = 1024

LEGAL_DOCUMENTS = [
    {
        "name": "Luat Phong, chong ma tuy 2021",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "urls": [
            "https://congbao.chinhphu.vn/tai-ve-van-ban-so-73-2021-qh14-33659-35651?format=pdf",
            "https://vbpl.vn/FileData/TW/Lists/vbpq/Attachments/152501/VanBanGoc_73_2021_QH14%20%281%29.pdf",
        ],
    },
    {
        "name": "Nghi dinh 105/2021/ND-CP huong dan Luat Phong, chong ma tuy",
        "filename": "nghi-dinh-105-2021.pdf",
        "urls": [
            "https://congbao.chinhphu.vn/tai-ve-van-ban-so-105-2021-nd-cp-34944-37821?format=pdf",
            "https://congbao.cdnchinhphu.vn/CongBaoCP/VanBan/2021/12/34944/37821-1-20211047-1048105-2021-nd-cp.pdf",
        ],
    },
    {
        "name": "Nghi dinh 57/2022/ND-CP ve danh muc chat ma tuy va tien chat",
        "filename": "nghi-dinh-57-2022.pdf",
        "urls": [
            "https://congbao.chinhphu.vn/tai-ve-van-ban-so-57-2022-nd-cp-37734-41623?format=pdf",
            "https://chinhphu.vn/?docid=206454&pageid=27160",
        ],
    },
]


def setup_directory() -> None:
    """Create data/landing/legal/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _download_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def _looks_like_download(content: bytes) -> bool:
    if len(content) <= MIN_FILE_SIZE_BYTES:
        return False
    prefix = content[:20].lower()
    return prefix.startswith(b"%pdf") or b"<html" not in prefix


def download_file(document: dict) -> Path:
    """Download one legal document, trying official URL first and mirrors next."""
    filepath = DATA_DIR / document["filename"]
    errors = []

    for url in document["urls"]:
        try:
            print(f"Downloading {document['name']} from {url}")
            content = _download_bytes(url)
            if not _looks_like_download(content):
                raise ValueError(f"Downloaded content is too small or looks like HTML: {len(content)} bytes")
            filepath.write_bytes(content)
            print(f"Saved: {filepath} ({filepath.stat().st_size} bytes)")
            return filepath
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not download " + document["filename"] + "\n" + "\n".join(errors))


def collect_legal_documents() -> list[Path]:
    """Download all configured legal documents."""
    setup_directory()
    downloaded = []
    for document in LEGAL_DOCUMENTS:
        downloaded.append(download_file(document))
    return downloaded


if __name__ == "__main__":
    files = collect_legal_documents()
    print(f"Done. Downloaded {len(files)} legal documents.")
