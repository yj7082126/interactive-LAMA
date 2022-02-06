# Tornado is more robust - consider using over Flask if do not need to worry about templates?
import argparse
import base64
from datetime import datetime
import numpy as np
import logging
from pathlib import Path
import socket
import sys
import signal
import yaml

import tornado.ioloop as ioloop
import tornado.web as web
import tornado.escape as escape

import cv2
from PIL import Image, ImageFilter
import torch

from saicinpainting.training.modules.ffc import FFCResNetGenerator

STATIC_FOLDER = Path("static")
STATIC_IMG_FOLDER = Path("img")

class UploadHandler(web.RequestHandler):
    def initialize(self, dir):
        self.dir = dir

    def get_img_from_req(self, body):
        data = base64.b64decode(str(body).split(",")[1])
        encoded_img = np.frombuffer(data, dtype = np.uint8)
        img_data = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_data)

    def get_mask_gray(self, img_data, filt=13):
        mask = img_data.convert("L").point(lambda p: p <= 0 and 255)
        mask = mask.filter(ImageFilter.ModeFilter(size=filt))
        return mask

    def get_mask_magenta(self, img_data, filt=13):
        mask = Image.new('L', img_data.size)
        d = img_data.getdata()
        new_d = []
        for item in d:
            if item[0] in range(240,256) and item[1] in range(0,16) and item[2] in range(240,256):
                new_d.append(255)
            else:
                new_d.append(0)
        mask.putdata(new_d)
        mask = mask.filter(ImageFilter.ModeFilter(size=filt))
        return mask

    def run_model(self, mask):
        img = cv2.imread(str(app.imgs[app.ind]))
        img = (np.asarray(img) / 255.).astype(np.float32)

        mask = mask.resize((img.shape[1], img.shape[0]), 3)
        mask = ((np.asarray(mask) / 255.) > 0.9).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        h, w, c = img.shape
        out_h = h if h % 8 == 0 else (h // 8 + 1) * 8
        out_w = w if w % 8 == 0 else (w // 8 + 1) * 8
        img_t  = np.pad(img,  ((0, out_h-h), (0, out_w-w), (0,0)), mode='symmetric')
        mask_t = np.pad(mask, ((0, out_h-h), (0, out_w-w), (0,0)), mode='symmetric')

        img_t = torch.from_numpy(img_t).permute(2,0,1).to(self.application.device)
        mask_t = torch.from_numpy(mask_t).permute(2,0,1).to(self.application.device)
        masked_img_t = img_t * (1 - mask_t)
        masked_img_t = torch.cat([masked_img_t, mask_t], 0).unsqueeze(0)

        with torch.no_grad():
            predicted_image = self.application.model(masked_img_t)

        inpaint = mask_t * predicted_image + (1 - mask_t) * img_t
        predict = inpaint[0].permute(1,2,0).cpu().numpy()
        predict = (predict * 255.).astype(np.uint8)[:,:,::-1]
        return Image.fromarray(predict)

    def post(self, name=None):
        self.application.logger.info("Recieved a file")
        img_name = app.imgs[app.ind].stem

        img_data = self.get_img_from_req(self.request.body)
        location = self.dir.joinpath(f"{img_name}.jpg")
        img_data.save(location)

        mask = self.get_mask_magenta(img_data)
        location2 = self.dir.joinpath(f"{img_name}_mask.jpg")
        mask.save(location2)

        predict = self.run_model(mask)
        location3 = self.dir.joinpath(f"{img_name}_predict.jpg")
        predict.save(location3)

        self.write({"result": "success", "location": str(location3)})

class ChangeHandler(web.RequestHandler):        
    def post(self):
        name = escape.to_basestring(self.request.body)
        app = self.application
        if name == "left":
            app.ind = app.ind - 1 if app.ind > 0 else app.ind
        else:
            app.ind = app.ind + 1 if app.ind < len(app.imgs) - 1 else app.ind
        location = app.imgs[app.ind]

        self.write({"result" : "success", "location" : str(location)})

class MainHandler(web.RequestHandler):
    def initialize(self, dir):
        self.dir = dir

    def get(self): 
        self.render("index.html", name=str(self.dir))

class MainApplication(web.Application):
    is_closing = False

    def signal_handler(self, signum, frame):
        logging.info('exiting...')
        self.is_closing = True

    def try_exit(self):
        if self.is_closing:
            ioloop.IOLoop.instance().stop()
            logging.info('exit success')

    def load_model(self, model_loc):
        model_loc = Path(model_loc)
        ckpt = torch.load(model_loc.joinpath("models/best.ckpt"), map_location="cpu")
        gen_weight = {k.replace("generator.", ""):v for k,v in ckpt["state_dict"].items() if k.startswith("generator.")}
        hparams = yaml.load(model_loc.joinpath("config.yaml").read_bytes(), Loader=yaml.FullLoader)["generator"]
        del hparams["kind"]

        model = FFCResNetGenerator(**hparams).eval().to(self.device)
        model.load_state_dict(gen_weight)
        print("Model loaded successfully")
        return model

    def __init__(self, args):
        web.Application.__init__(self, **vars(args))
        self.port = args.port
        self.imgdir = args.imgdir
        self.address = args.address
        self.device = args.device
        self.template_path = "templates"

        self.ioloop = ioloop.IOLoop.instance()
        self.logger = logging.getLogger()

        self.sess_name = datetime.now().strftime('%y-%m-%d_%H:%M:%S:%f')[:-3]
        self.dir = Path("img/webui").joinpath(self.sess_name)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.imgs = list(Path(self.imgdir).glob("*.jpg"))
        self.imgs += list(Path(self.imgdir).glob("*.png"))
        self.imgs = sorted(self.imgs)
        self.ind = 0

        self.model = self.load_model(args.model_loc)
        
        self.add_handlers(".*", [
            (r"/", MainHandler, {"dir" : self.imgs[self.ind]}),
            (r"/upload", UploadHandler, {"dir" : self.dir}),
            (r"/change", ChangeHandler, {}),
            (r"/cartoon_data/(.*)", web.StaticFileHandler,   {"path": "/datasets/RD/cartoon_data"}),
            (r"/img/(.*)", web.StaticFileHandler,   {"path": "img"}),
            (r".*/static/(.*)", web.StaticFileHandler,  {"path": "static"})
        ])

    def run(self):
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            self.listen(self.port, self.address)
            ioloop.PeriodicCallback(self.try_exit, 100).start()

        except socket.error as e:
            self.logger.fatal(f"Unable to listen on {self.address}:{self.port} = {e}")
            sys.exit(1)
        self.ioloop.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="127.0.0.1", help='Url')
    parser.add_argument('--port', default=6006, help='Port to listen on.')
    parser.add_argument('--imgdir', type=str, default="img/inpainting_examples")
    parser.add_argument('--model_loc', type=str, default="checkpoints/big-lama")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--template_path', default="templates", help='Path to templates')
    args = parser.parse_args()

    app = MainApplication(args)
    app.run()
