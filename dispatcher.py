# -*- coding: utf-8 -*-

# 后台调度程序，异步执行，使用redis作为消息队列

import sys, json, time
import concurrent.futures
from datetime import datetime
import binascii

from utils import helper
from utils import logger
from settings import REDIS_CONFIG, MAX_DISPATCHER_WORKERS

import ocr

logger = logger.get_logger(__name__)

ocr_model = None


def process_api(request_id, request_msg):
    request = request_msg
    try:
        if request['api']=='/api/ocr/det': # 文本检测
            # base64 图片 转为 opencv 数据
            img, shape = ocr.load_image_b64(request['params']['image'])
            r1, _ = ocr_model.detect(img)

            # [x_min, x_max, y_min, y_max]
            boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in r1] # 屏蔽 numpy.int64

            # 准备结果
            result = { 'code' : 0, 'msg':'success', 'boxes' : boxes, 'shape' : shape }

        elif request['api']=='/api/ocr/rec': # 文字识别
            # base64 图片 转为 opencv 数据
            img, _ = ocr.load_image_b64(request['params']['image'])
            param_boxes = json.loads(request['params']['boxes'])
            #  param_boxes 格式 [ [x_min, x_max, y_min, y_max] ]
            r1 = ocr_model.recognize(img, param_boxes, [])

            # 准备结果
            result = { 'code' : 0, 'msg':'success', 'result' : r1[0][1] }

        elif request['api']=='/api/ocr/ocr': # 文本 OCR
            # base64 图片 转为 opencv 数据
            img, _ = ocr.load_image_b64(request['params']['image'])
            r1 = ocr_model.ocr_w_merge(img)

            # [x_min, x_max, y_min, y_max]
            result = [[
                    [
                        min(int(i[0][0][0]), int(i[0][3][0])),
                        max(int(i[0][1][0]), int(i[0][2][0])),
                        min(int(i[0][0][1]), int(i[0][1][1])),
                        max(int(i[0][2][1]), int(i[0][3][1])),
                    ],
                    i[1],
                    i[2]
                ] for i in r1] # 屏蔽 numpy.int64

            # 准备结果
            result = { 'code' : 0, 'msg':'success', 'result' : result }

        else: # 未知 api
            logger.error('Unknown api: '+request['api']) 
            result = { 'code' : 9900, 'msg' : '未知 api 调用' }

    except binascii.Error as e:
        logger.error("编码转换异常: %s" % e)
        result = { 'code' : 9901, 'msg' : 'base64编码异常: '+str(e) }

    except json.decoder.JSONDecodeError as e:
        logger.error("json转换异常: %s" % e)
        result = { 'code' : 9902, 'msg' : 'json编码异常: '+str(e) }

    except Exception as e:
        logger.error("未知异常: %s" % e, exc_info=True)
        result = { 'code' : 9998, 'msg' : '未知错误: '+str(e) }

    return result



def process_thread(msg_body):
    try:

        logger.info('{} Calling api: {}'.format(msg_body['request_id'], msg_body['data'].get('api', 'Unknown'))) 

        start_time = datetime.now()

        api_result = process_api(msg_body['request_id'], msg_body['data'])

        logger.info('1 ===> [Time taken: {!s}]'.format(datetime.now() - start_time))
        
        # 发布redis消息
        helper.redis_publish(msg_body['request_id'], api_result)
        
        logger.info('{} {} [Time taken: {!s}]'.format(msg_body['request_id'], msg_body['data']['api'], datetime.now() - start_time))

        sys.stdout.flush()

    except Exception as e:
        logger.error("process_thread异常: %s" % e, exc_info=True)



if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: dispatcher.py <QUEUE_NO.>")
        sys.exit(2)

    queue_no = sys.argv[1]
    #gpu = sys.argv[2]

    print('Request queue NO. ', queue_no)

    ocr_model = ocr.OCR(recognizer=False) # 只使用文本检测

    sys.stdout.flush()

    while 1:
        try:
            # redis queue
            ps = helper.redis_subscribe(REDIS_CONFIG['REQUEST-QUEUE']+queue_no)

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DISPATCHER_WORKERS) # 建议与cpu核数相同

            for item in ps.listen():        #监听状态：有消息发布了就拿过来
                logger.info('reveived: type=%s running=%d pending=%d'% \
                    (item['type'], len(executor._threads), executor._work_queue.qsize())) 
                if item['type'] == 'message':
                    #print(item)
                    msg_body = json.loads(item['data'].decode('utf-8'))

                    future = executor.submit(process_thread, msg_body)
                    logger.info('Thread future: '+str(future)) 

                sys.stdout.flush()

        except Exception as e:
            logger.info('Exception: '+str(e)) 
            time.sleep(20)
