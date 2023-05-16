from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from retrieval_test import retrieve, calcDescriptors
import trimesh
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory='templates')
origins = [
    "http://106.15.224.125:8080",
    "http://106.15.224.125:8041",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

descriptor_set = np.load('C:/Users/Admin/CAD_parts/descriptors.npy')
names = np.load('C:/Users/Admin/CAD_parts/names.npy')
codebook = np.load(r'C:/Users/Admin/CAD_parts/codebook.npy')


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    try:
        file_location = f"./stl_base/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        return JSONResponse(status_code=200, content={"message": "File uploaded successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post('/retrieve')
def retrieve_serve(request: UploadFile):
    mesh = trimesh.load_mesh(UploadFile)
    d2, bof_desc = calcDescriptors(mesh, codebook)


    return retrieve(request["input_name"], request["output_number"]).tolist()[1:]


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='106.15.224.125', port=8041)
