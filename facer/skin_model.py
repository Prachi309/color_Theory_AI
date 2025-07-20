import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import gc
import traceback

class LazySkinModel:
    _instance = None
    _model = None
    _transform = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LazySkinModel, cls).__new__(cls)
        return cls._instance
    
    def _load_model(self):
        if not self._loaded:
            print("Loading skin model (lazy loading)...")
            gc.collect()
            
            model = models.resnet18(pretrained=True)
            num_classes = 4
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

            import os
            model_path = os.path.join(os.path.dirname(__file__), "cp/best_model_resnet_ALL.pth")
            print("Loading model from:", model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self._model = models.resnet18(pretrained=True)
            self._model.fc = nn.Linear(in_features, num_classes)
            self._model.load_state_dict(state_dict)

            self._transform = transforms.Compose([
                transforms.Resize((160, 160)),  
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            self._model.eval()
            self._loaded = True
            print("Skin model loaded successfully!")
            
            del model
            gc.collect()
    
    def predict(self, img):
        self._load_model()
        from PIL import Image
        import cv2
        import numpy as np
        cv_img = cv2.imread(img)
        if cv_img is not None:
            cv_img = cv2.resize(cv_img, (160, 160), interpolation=cv2.INTER_AREA)
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv_img_rgb)
        else:
            image = Image.open(img).convert('RGB')
            image = image.resize((160, 160), Image.Resampling.LANCZOS)
        
        image = self._transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self._model(image)
        pred_index = output.argmax().item()
        print("Decided color: ", pred_index)
        del image, output
        gc.collect()
        
        return pred_index

_lazy_model = None

def get_season(img):
    global _lazy_model
    if _lazy_model is None:
        _lazy_model = LazySkinModel()
    return _lazy_model.predict(img)
