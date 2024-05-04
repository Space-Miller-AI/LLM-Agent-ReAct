from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader


def load_pdf(file_name, method="unstructured"):
    if method == "unstructured":
        loader = UnstructuredPDFLoader(file_name)
        data = loader.load()
    elif method == "pypdf":
        loader = PyPDFLoader(file_name)
        data = loader.load_and_split()
    else:
        raise ValueError("Method not supported: {}".format(method))

    return data
