from paperqa import Docs

docs = Docs()
response = docs.query('What is AI?')
print(type(response))