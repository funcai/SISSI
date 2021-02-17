from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-german")
print(tokenizer("morgen"))
print(tokenizer("heute"))
print(tokenizer("übermorgen"))
print(tokenizer("Kaufhaus"))
print(tokenizer("Apfel"))