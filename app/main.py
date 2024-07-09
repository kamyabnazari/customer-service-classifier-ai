from fastapi import FastAPI
from app.routers import root, predict_routes
from app.dependencies import configure_cors, lifespan

app = FastAPI(title="CustomerServiceClassifierAI Project", lifespan=lifespan)

# Configure CORS
configure_cors(app)

app.include_router(root.router)
app.include_router(predict_routes.router, prefix="/api/predict")