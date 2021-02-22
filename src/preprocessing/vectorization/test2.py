from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')
tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')

FACT_TO_GENERATE_QUESTION_FROM = "Bread can be spread with butter, dipped into liquids such as gravy, olive oil, or soup;[28] it can be topped with various sweet and savory spreads, or used to make sandwiches containing meats, cheeses, vegetables, and condiments."

inputs = tokenizer([FACT_TO_GENERATE_QUESTION_FROM], return_tensors='pt')

# Generate Summary
question_ids = model.generate(inputs['input_ids'], num_beams=5, early_stopping=True)
print(tokenizer.batch_decode(question_ids, skip_special_tokens=True))