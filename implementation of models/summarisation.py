from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration
def summary(text):
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name) 
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output = model.generate(
        input_ids, 
        max_length=32, 
        num_beams=5, 
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
