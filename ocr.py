import sys
import json
import base64
from io import BytesIO
import numpy as np
import cv2
import easyocr
from utils import link_x_boxes


class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


# 将 base64 编码的图片转为 opencv 数组
def load_image_b64(b64_data, remove_color=True, max_size=None):
    data = base64.b64decode(b64_data) # Bytes
    tmp_buff = BytesIO()
    tmp_buff.write(data)
    tmp_buff.seek(0)
    file_bytes = np.asarray(bytearray(tmp_buff.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if remove_color:
        img = img[:, :, ::-1] # 去色，强化图片
    tmp_buff.close()
    # 压缩处理
    if max_size is not None:
        max_width = max(img.shape)
        if max_width>max_size: # 图片最大宽度为 1500
            ratio = max_size/max_width
            img = cv2.resize(img, (round(img.shape[1]*ratio), round(img.shape[0]*ratio)))
    return img, (img.shape[1], img.shape[0])


class OCR():
    def __init__(self, device="cuda:0"):
        # this needs to run only once to load the model into memory
        self.reader = easyocr.Reader(['ch_sim','en'], gpu=device)
        #self.reader = easyocr.Reader(['ch_sim','en'], detect_network = 'dbnet50', gpu=True)
        print("Load OCR model to device:", device)

    def detect(self, img):
        height, width, channel = img.shape
        print(img.shape)
        horizontal_list, free_list = self.reader.detect(img, canvas_size=max(height, width))
        return horizontal_list[0], free_list[0]

    def recognize(self, img, horizontal_list, free_list):
        result = self.reader.recognize(img, horizontal_list, free_list)
        return result

    def ocr(self, img):    
        result = self.reader.readtext(img)
        return result

    def ocr_w_merge(self, img):
        h, w, c = img.shape

        det_boxes, _ = self.detect(img)

        # 转换坐标： (x_min, x_max, y_min, y_max) --> (x1, y1, x2, y2)
        rects = [ [ i[0], i[2], i[1], i[3] ] for i in det_boxes]

        # 横向合并 box
        connector = link_x_boxes.BoxesConnector(rects, w, max_dist=15, overlap_threshold=0.3)
        new_boxes = connector.connect_boxes()

        # 转换坐标/ 过滤掉 高大于宽 的 box 
        det_boxes = [ [ i[0], i[2], i[1], i[3] ] for i in new_boxes if (i[2]-i[0]) > (i[3]-i[1]) ]

        result = self.recognize(img, det_boxes, [])
        return result


def draw_boxes(img, boxes):
    h, w, c = img.shape

    for i, box in enumerate(boxes):
        cv2.polylines(img, [np.array(box, np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

    cv2.imwrite('boxes.jpg',img)


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python3 %s <image_path>" % sys.argv[0])
        sys.exit(2)

    img = cv2.imread(sys.argv[1])

    ocr_model = OCR()

    
    # 一步 ocr
    r3 = ocr_model.ocr_w_merge(img)

    boxes = [i[0] for i in r3]
    '''

    # 只检测文本
    r1, r2 = ocr_model.detect(img)

    print(r1)  # [x_min, x_max, y_min, y_max]
    print(r2)
    
    # box转换为 4个点坐标
    # [x_min, x_max, y_min, y_max] --> [左上, 右上, 右下, 左下] 
    boxes = [[[i[0],i[2]],[i[1],i[2]],[i[1],i[3]],[i[0],i[3]]] for i in r1]

    #r3 = recognize(img, r1, r2)

    # [左上, 右上, 右下, 左下] --> [x_min, x_max, y_min, y_max]
    box = boxes[0]
    box = [box[0][0], box[1][0], box[0][1], box[2][1]]
    print(box)
    r3 = ocr_model.recognize(img, [box], [])
    '''

    print(boxes)
    draw_boxes(img, boxes)

    for i in r3:
        print(f"{i[2]:.6f}\t{i[1]}")


    json.dump(
        r3,
        open('result.json', 'w', encoding='utf-8'),
        #indent=4,
        ensure_ascii=False,
        cls=JsonEncoder
    )
