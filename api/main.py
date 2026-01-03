import logging
import sys
import asyncio # Import asyncio
from pathlib import Path
from fastapi import FastAPI, Request

# Add project root to path to allow absolute imports
project_root = Path(__file__).resolve().parents[1] # api/main.py -> project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import routers and other necessary components
from api.routers import pipelines, server, results, he300
from utils.logging_config import setup_logging
from utils.concurrency_monitor import ConcurrencyMonitor
from core.engine import EthicsEngine # Import the engine class
from config.settings import settings

# Setup logging based on config
setup_logging()
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI(
    title="Ethics Engine Enterprise API",
    description="API for managing and executing Ethics Engine pipelines.",
    version="0.1.0" # Consider pulling from a config or __version__
)

# --- Initialize and Store Shared State ---
# Initialize shared components
llm_limit = settings.max_concurrent_llm_calls
llm_semaphore = asyncio.Semaphore(llm_limit) # Create the semaphore
concurrency_monitor = ConcurrencyMonitor(semaphore=llm_semaphore, limit=llm_limit) # Pass semaphore and limit
ethics_engine = EthicsEngine() # Instantiate the engine (loads configs by default)

# Store shared components in app state
app.state.concurrency_monitor = concurrency_monitor
app.state.ethics_engine = ethics_engine # Store the engine instance
app.state.llm_semaphore = llm_semaphore # Store semaphore if needed elsewhere (e.g., LLM handler)
logger.info(f"Concurrency monitor initialized with limit: {llm_limit}")
logger.info("EthicsEngine instance created and configurations loaded.")

# --- Middleware (Optional Example: Logging requests) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# --- Include Routers ---
app.include_router(pipelines.router)
app.include_router(server.router)
app.include_router(results.router)
app.include_router(he300.router)  # HE-300 benchmark integration

# --- Root Endpoint ---

@app.get("/", tags=["General"])
async def read_root():
    """Provides a simple welcome message."""
    return {"message": "Welcome to the Ethics Engine Enterprise API"}

@app.get("/health", tags=["General"])
async def health_check():
    """Provides a health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}

# --- Application Startup/Shutdown Events (Optional Example) ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Ethics Engine Enterprise API...")
    # Potentially initialize other resources here

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Ethics Engine Enterprise API...")
    # Potentially clean up resources here
