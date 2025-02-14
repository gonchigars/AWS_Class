from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the create_bucket function from Create_bucket.py.
# Ensure that Create_bucket.py is in the same directory or adjust the import accordingly.
from Create_bucket import create_bucket

app = FastAPI()

# Define a request model for the bucket creation endpoint.
class BucketRequest(BaseModel):
    bucket_name: str

@app.post("/bucket")
async def create_bucket_endpoint(request: BucketRequest):
    # Validate that the bucket name is provided.
    if not request.bucket_name:
        raise HTTPException(status_code=400, detail="Bucket name cannot be empty")
    
    # Attempt to create the bucket using the create_bucket function.
    success = create_bucket(request.bucket_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create bucket")
    
    return {"message": f"Bucket '{request.bucket_name}' created successfully!"}

# Run the application using uvicorn if this script is executed directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)