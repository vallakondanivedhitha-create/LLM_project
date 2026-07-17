#Hugging Face Model to Fine-tune
from transformers import AutoModelForCausalLM, AutoTokenizer

#Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

#Take user input for fine-tuning
text = "What is the Capital of France?"

#Tokenize the input text 
inputs = tokenizer(text,return_tensors="pt")

#Display the tokenized input
print("Tokenized input:")
print(tokenizer.tokenize(text))
print(inputs["input_ids"])


#Attention
print(inputs["attention_mask"])

#Convert the input ids back to text
tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0])
print("Tokens:",tokens)

#Run the Model
outputs = model(**inputs)


print("Model Outputs :", outputs)

print("Logits :",outputs.logits)
print("Logits shape :",outputs.logits.shape)
print("vocab size:",tokenizer.vocab_size)

total_scores = outputs.logits[0,-1,:]
print("Total_Scores",total_scores)
