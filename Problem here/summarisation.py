# Model 1:
# from transformers import pipeline
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration


# def summary(text):

#     summarizer = pipeline("summarization", model="lidiya/bart-large-xsum-samsum")
#     conversation = text
#     return summarizer(conversation)

    # Example:
    # conversation = '''Hannah: Hey, do you have Betty's number?
    # Amanda: Lemme check
    # Amanda: Sorry, can't find it.
    # Amanda: Ask Larry
    # Amanda: He called her last time we were at the park together
    # Hannah: I don't know him well
    # Amanda: Don't be shy, he's very nice
    # Hannah: If you say so..
    # Hannah: I'd rather you texted him
    # Amanda: Just text him 🙂
    # Hannah: Urgh.. Alright
    # Hannah: Bye
    # Amanda: Bye bye                                       
    # # '''

# Model 2:
def summary(text):
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name) # If you want to use the Tensorflow model 
                                                                        # just replace with TFPegasusForConditionalGeneration
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output = model.generate(
        input_ids, 
        max_length=32, 
        num_beams=5, 
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
 # Some text to summarize here
    # text_to_summarize = "National Commercial Bank (NCB), Saudi Arabia’s largest lender by assets, agreed to buy rival Samba Financial Group for $15 billion in the biggest banking takeover this year.NCB will pay 28.45 riyals ($7.58) for each Samba share, according to a statement on Sunday, valuing it at about 55.7 billion riyals. NCB will offer 0.739 new shares for each Samba share, at the lower end of the 0.736-0.787 ratio the banks set when they signed an initial framework agreement in June.The offer is a 3.5% premium to Samba’s Oct. 8 closing price of 27.50 riyals and about 24% higher than the level the shares traded at before the talks were made public. Bloomberg News first reported the merger discussions.The new bank will have total assets of more than $220 billion, creating the Gulf region’s third-largest lender. The entity’s $46 billion market capitalization nearly matches that of Qatar National Bank QPSC, which is still the Middle East’s biggest lender with about $268 billion of assets."   
# Generated Output: Saudi bank to pay a 3.5% premium to Samba share price. Gulf region’s third-largest lender will have total assets of $220 billion

