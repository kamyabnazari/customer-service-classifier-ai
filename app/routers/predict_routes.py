from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import get_database
from app.utils.predict_utils import insert_prediction, generate_joke

router = APIRouter()
database = get_database()

@router.post("/generate-prediction/")
async def get_joke(name: str, database: database = Depends(get_database)):
    joke_prompt = "Tell me a prediction about artificial intelligence."
    try:
        result = generate_joke(joke_prompt)
        if 'error' in result:
            raise Exception(result['error'])

        joke_text = result.get('choices')[0].get('text', 'No prediction generated.')
        await insert_prediction(database, name, joke_prompt, joke_text, "Completed")
        return {
            "name": name,
            "request": joke_prompt,
            "response": joke_text,
            "status": "Completed"
        }
    except Exception as e:
        error_message = str(e)
        await insert_prediction(database, name, joke_prompt, error_message, "Failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate joke: {error_message}")