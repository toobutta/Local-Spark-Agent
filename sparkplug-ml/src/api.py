"""
FastAPI application for SparkPlug ML Pipeline
Provides REST endpoints for ML operations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import logging

from .utilities.configuration_manager import ConfigurationManager
from .utilities.version_control import VersionControlManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SparkPlug ML Pipeline API",
    description="REST API for SparkPlug machine learning operations",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
config_manager = ConfigurationManager("configs")
version_manager = VersionControlManager("data/versioning")

# Pydantic models for request/response
class PipelineRequest(BaseModel):
    pipeline_name: str
    config_overrides: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    model_name: str
    dataset_path: str
    config: Dict[str, Any]

class ExperimentRequest(BaseModel):
    name: str
    config: Dict[str, Any]

class ModelVersionResponse(BaseModel):
    version_id: str
    model_name: str
    metadata: Dict[str, Any]
    created_at: str

# Background tasks storage (in production, use proper task queue)
running_tasks: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SparkPlug ML Pipeline API", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/pipeline/run")
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Run a complete ML pipeline"""
    try:
        # Load pipeline configuration
        pipeline_config = config_manager.get_pipeline_config(request.pipeline_name)

        # Merge with overrides
        if request.config_overrides:
            final_config = config_manager.merge_configs(
                pipeline_config.__dict__,
                request.config_overrides
            )
        else:
            final_config = pipeline_config.__dict__

        # Generate task ID
        task_id = f"pipeline_{request.pipeline_name}_{asyncio.get_event_loop().time()}"

        # Start pipeline in background
        background_tasks.add_task(run_pipeline_background, task_id, final_config)

        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Pipeline {request.pipeline_name} started"
        }

    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/status/{task_id}")
async def get_pipeline_status(task_id: str):
    """Get status of a pipeline task"""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return running_tasks[task_id]

@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    try:
        task_id = f"training_{request.model_name}_{asyncio.get_event_loop().time()}"

        # Start training in background
        background_tasks.add_task(run_training_background, task_id, request)

        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Training {request.model_name} started"
        }

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiment/create")
async def create_experiment(request: ExperimentRequest):
    """Create a new experiment"""
    try:
        experiment_id = version_manager.create_experiment(request.name, request.config)

        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": f"Experiment {request.name} created"
        }

    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/versions")
async def list_model_versions(model_name: str):
    """List all versions of a model"""
    try:
        versions = version_manager.list_model_versions(model_name)
        return {"model_name": model_name, "versions": versions}

    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/versions/{version_id}")
async def get_model_version(model_name: str, version_id: str):
    """Get a specific model version"""
    try:
        version = version_manager.get_model_version(model_name, version_id)
        return version

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model version not found")
    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
async def list_experiments():
    """List all experiments"""
    try:
        # For now, return a placeholder
        # In production, implement experiment listing
        return {"experiments": []}

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task implementations
async def run_pipeline_background(task_id: str, config: Dict[str, Any]):
    """Run pipeline in background"""
    try:
        running_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Initializing pipeline..."
        }

        # Simulate pipeline execution
        steps = [
            ("data_loading", "Loading data..."),
            ("preprocessing", "Preprocessing data..."),
            ("training", "Training model..."),
            ("evaluation", "Evaluating model..."),
            ("deployment", "Deploying model...")
        ]

        for i, (step, message) in enumerate(steps):
            running_tasks[task_id].update({
                "progress": (i + 1) * 20,
                "message": message
            })

            # Simulate work
            await asyncio.sleep(2)

        running_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Pipeline completed successfully"
        })

    except Exception as e:
        running_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Pipeline failed: {str(e)}"
        })

async def run_training_background(task_id: str, request: TrainingRequest):
    """Run training in background"""
    try:
        running_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Starting training..."
        }

        # Simulate training process
        for epoch in range(1, 11):
            progress = epoch * 10
            running_tasks[task_id].update({
                "progress": progress,
                "message": f"Training epoch {epoch}/10..."
            })

            # Simulate training time
            await asyncio.sleep(1)

        running_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Training completed successfully"
        })

    except Exception as e:
        running_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
