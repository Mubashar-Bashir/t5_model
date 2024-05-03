from fastapi import FastAPI
from fastapi.responses import FileResponse
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

# Default route
@app.get("/")
def hello():
    return "Hello FastAPI for Model training"


# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("T5_MODEL/model")
model = T5ForConditionalGeneration.from_pretrained("T5_MODEL/model")

# @app.get("/")
# def read_index():
#     return FileResponse("index.html")

# Route to translate text using the T5 model
@app.post("/translate")
async def translate_text(text: str):
    output = model.generate(
        input_ids=tokenizer.encode(text, return_tensors="pt"),
        max_length=200,
        num_beams=5,
        early_stopping=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    output_text = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    return {"translated_text": output_text}

# Route for a sample data endpoint
@app.get("/sample-data/")
async def sample_data():
    return {
        "text": "translate English to French: Hello, how are you?",
        "expected_translation": "Bonjour, comment vas-tu ?"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)