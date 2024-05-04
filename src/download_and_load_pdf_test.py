from download_pdf import download_pdf
from load_pdf import load_pdf

if __name__ == "__main__":
    test_url = "https://arxiv.org/pdf/2212.12934.pdf"
    download_pdf(test_url)
    data = load_pdf("temp.pdf")
    print(data)
    print()
