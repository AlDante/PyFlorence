import pypdfium2 as pdfium

EBUDIR="/Users/david/Downloads/Bridge/EBU/"
EBUNAME="1946-09"
EBUPDFSUFFIX= ".pdf"
EBUPDFPATH= EBUDIR + EBUNAME + EBUPDFSUFFIX

EBUPNGSUFFIX= ".png"
EBUPNGPATH= EBUDIR + EBUNAME + EBUPNGSUFFIX

if __name__ == "__main__":
    pdf = pdfium.PdfDocument(EBUPDFPATH)
    version = pdf.get_version()  # get the PDF standard version
    n_pages = len(pdf)  # get the number of pages in the document
    page = pdf[11]  # load a page

    bitmap = page.render(
        scale = 1,    # 72dpi resolution
        rotation = 0, # no additional rotation
        # ... further rendering options
    )
    pil_image = bitmap.to_pil()
    pil_image.show()
    pil_image.save(EBUPNGPATH)
