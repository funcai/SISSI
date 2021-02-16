from SpacyVectorizer import SpacyVectorizer

v = SpacyVectorizer()
print(v.vectorize("Wo die wohligen Wonnen wallen"))
print(v.getOOVWords("Wo die wurzkrugen Wonnen wallen"))