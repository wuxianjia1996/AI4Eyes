from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model import EyeModel
import torchvision.transforms as transforms
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
eye_model = EyeModel()
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def upload_image(image: UploadFile = File(...)):
    try:
        # 读取图片内容
        image_data = await image.read()

        # 将图片内容转换为PIL.Image格式
        image_pil = Image.open(io.BytesIO(image_data))
        # print(image_pil)

        # 在这里处理PIL图像对象，例如进行图片处理或分析
        img_tensor = transformer(image_pil)
        # 这里不能太小否则报错

        # print(img_tensor)
        probs, preds = eye_model.predict2(img_tensor)

        return JSONResponse(status_code=200,
                            content={"message": "Image uploaded successfully", "probs": probs.tolist()[0],
                                     "preds": preds.tolist()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})


@app.post("/predict2")
async def upload_image(image: UploadFile = File(...)):
    try:
        # 读取图片内容
        image_data = await image.read()

        # 将图片内容转换为PIL.Image格式
        image_pil = Image.open(io.BytesIO(image_data))
        # print(image_pil)

        # 在这里处理PIL图像对象，例如进行图片处理或分析
        img_tensor = transformer(image_pil)
        # 如果是RGBA 就会报错
        # 这里不能太小否则报错

        # print(img_tensor)
        probs, preds = eye_model.predict2(img_tensor)

        return JSONResponse(status_code=200,
                            content={"message": "Image uploaded successfully", "probs": probs.tolist()[0],
                                     "preds": preds.tolist()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})


@app.post("/predict3")
async def upload_image(image: UploadFile = File(...)):
    try:
        # 读取图片内容
        image_data = await image.read()

        # 将图片内容转换为PIL.Image格式
        image_pil = Image.open(io.BytesIO(image_data))
        # print(image_pil)

        # 在这里处理PIL图像对象，例如进行图片处理或分析
        img_tensor = transformer(image_pil)
        # 这里不能太小否则报错

        # print(img_tensor)
        probs, preds = eye_model.predict3(img_tensor)

        return JSONResponse(status_code=200,
                            content={"message": "Image uploaded successfully", "probs": probs.tolist()[0],
                                     "preds": preds.tolist()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})


@app.post("/predict4")
async def upload_image(image: UploadFile = File(...)):
    try:
        # 读取图片内容
        image_data = await image.read()

        # 将图片内容转换为PIL.Image格式
        image_pil = Image.open(io.BytesIO(image_data))
        # print(image_pil)

        # 在这里处理PIL图像对象，例如进行图片处理或分析
        img_tensor = transformer(image_pil)
        # 这里不能太小否则报错

        # print(img_tensor)
        probs, preds = eye_model.predict4(img_tensor)

        return JSONResponse(status_code=200,
                            content={"message": "Image uploaded successfully", "probs": probs.tolist()[0],
                                     "preds": preds.tolist()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10001)
