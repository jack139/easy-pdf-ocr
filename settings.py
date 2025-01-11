# mongo DB 配置
mongodb_ip = "127.0.0.1"
mongodb_cname = "rag_db"
mongodb_user = "ipcam"
mongodb_passwd = "ipcam"

# dispatcher 中 最大线程数
MAX_DISPATCHER_WORKERS = 8

############# 消息中间件设置

REDIS_CONFIG = {
    'SERVER' : '127.0.0.1',
    'PORT'   : '7480',
    'PASSWD' : 'e18ffb7484f4d69c2acb40008471a71c',
    'REQUEST-QUEUE' : 'EasyOCR-synchronous-asynchronous-queue',
    'REQUEST-QUEUE-NUM' : 1,
    'MESSAGE_TIMEOUT' : 60, # 结果返回消息超时，单位：秒
}
