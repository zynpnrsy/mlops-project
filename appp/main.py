
from fastapi import FastAPI

app = FastAPI(title="Telco Churn Model API")

@app.get("/")
def health_check():
    return {"status": "ok"}
