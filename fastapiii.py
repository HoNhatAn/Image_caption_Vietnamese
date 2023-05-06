from fastapi import FastAPI, File, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from fastapi.responses import FileResponse
import requests
from sample import run_inference,load_image,runn
from build_vocab import Vocabulary
app = FastAPI()
origins = ["*"]
# "*" means allow all origins, you can change it to a specific domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image.save("AnhDaLuu.jpg")
    image = Image.open("./AnhDaLuu.jpg")
    strr=runn(image)
    return {"filename": strr}
#
#
#
#
#
# import pickle
# from fastapi import FastAPI
#
# app = FastAPI()
#
# # Load pickled data
# with open('preprocessed_vocab_.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# # Define endpoint
# @app.get('/data')
# async def get_data():
#     return data
# @app.get("/upload")
# async def upload_file():
#     image = Image.open("./AnhDaLuu.jpg")
#     strr=runn(image)
#     return {"filename": strr}