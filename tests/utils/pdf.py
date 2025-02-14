from io import BytesIO
from typing import List

from pypdf import PageObject, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def _escape_pdf_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _create_page_with_text(
    text: str, width: int = 612, height: int = 792
) -> PageObject:
    page = PageObject.create_blank_page(width=width, height=height)

    content_stream = "BT\n"
    content_stream += "/F1 12 Tf\n"

    x = 72
    y = height - 72

    for line in text.splitlines():
        content_stream += f"{x} {y} Td\n"
        content_stream += f"({_escape_pdf_string(line)}) Tj\n"
        y -= 14
    content_stream += "ET\n"

    stream = DecodedStreamObject()
    stream.set_data(content_stream.encode("utf-8"))
    page[NameObject("/Contents")] = stream

    resources = DictionaryObject()
    font_dict = DictionaryObject()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_dict[NameObject("/F1")] = font
    resources[NameObject("/Font")] = font_dict
    page[NameObject("/Resources")] = resources

    return page


def gen_pdf(pages: List[str]) -> bytes:
    """
    Generate a PDF from a list of page strings.
    """
    writer = PdfWriter()

    for text in pages:
        page = _create_page_with_text(text)
        writer.add_page(page)

    output = BytesIO()
    writer.write(output)
    pdf_bytes = output.getvalue()
    output.close()

    return pdf_bytes
