# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="google-t5/t5-small")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

# Define directory paths
tokenizer_dir = "../T5_MODEL_PATH/model"
model_dir = "../T5_MODEL_PATH/model"

# Save tokenizer
tokenizer.save_pretrained(tokenizer_dir)

# Save model
model.save_pretrained(model_dir)
