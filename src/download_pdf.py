import os

import requests


def sanitize_url(url: str) -> str:
    """Sanitize the URL to make it easier to download."""
    if not url.endswith(".pdf"):
        if "ar5iv.labs.arxiv.org" in url:
            url = url.replace("ar5iv.labs.arxiv.org", "arxiv.org")
        url = url.replace("abs", "pdf")
        url = url.replace("html", "pdf")
        url += ".pdf"
    return url


def get_filename_from_url(url: str) -> str:
    return f"../{url.split('/')[-1]}"


def download_pdf(url: str):
    """Download a PDF from a URL."""
    url = sanitize_url(url)
    filename = get_filename_from_url(url)
    # If file already exists, don't download it again
    if os.path.exists(filename):
        # print(f"File {filename} already exists. Skipping download.")
        return filename
    response = requests.get(url)

    with open(filename, "wb") as file:
        file.write(response.content)
    return filename


# test_url = "https://arxiv.org/abs/2212.12934"
# test_url = "https://arxiv.org/pdf/2212.12934.pdf"
# print(download_pdf(test_url))
