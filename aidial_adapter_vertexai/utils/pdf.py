import asyncio
from io import BytesIO

from pypdf import PdfReader


async def get_pdf_page_count(doc: bytes) -> int:
    loop = asyncio.get_running_loop()

    def _sync_get_page_count():
        pdf_bytes_io = BytesIO(doc)
        pdf = PdfReader(pdf_bytes_io)
        return len(pdf.pages)

    num_pages = await loop.run_in_executor(
        None, lambda _: _sync_get_page_count()
    )

    return num_pages
