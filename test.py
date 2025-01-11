import sys
import json
import numpy as np
import cv2
import easyocr

# this needs to run only once to load the model into memory
reader = easyocr.Reader(['ch_sim','en'])
#reader = easyocr.Reader(['ch_sim','en'], detect_network = 'dbnet50')


class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def ocr(img):    
    result = reader.readtext(img)
    return result

def detect(img):
    horizontal_list, free_list = reader.detect(img)
    return horizontal_list[0], free_list[0]

def recognize(img, horizontal_list, free_list):
    result = reader.recognize(img, horizontal_list, free_list)
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

    '''
    # 一步 ocr
    r3 = ocr(img)

    boxes = [i[0] for i in r3]
    '''

    # 只检测文本
    r1, r2 = detect(img)

    print(r1)
    print(r2)
    
    boxes = [[[i[0],i[2]],[i[1],i[2]],[i[1],i[3]],[i[0],i[3]]] for i in r1]

    r3 = recognize(img, r1, r2)


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
