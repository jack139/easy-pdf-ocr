import numpy as np


def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


class BoxesConnector(object):
    def __init__(self, rects, imageW, max_dist=None, overlap_threshold=None):
        #print('max_dist',max_dist)
        #print('overlap_threshold',overlap_threshold )
        self.rects = np.array(rects)
        self.imageW = imageW
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)
        #print(self.r_index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        #print('rect',rect)

        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            #print('left',left)
            for idx in self.r_index[left]:
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:
                    return idx
        return -1

    def sub_graphs_connected(self):
        sub_graphs = []       #相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any(): #优先级是not > and > or
                v = index
                sub_graphs.append([v])
                # 级联多个框(大于等于2个)
                #print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    sub_graphs[-1].append(v)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):
            proposal = self.get_proposal(idx)
            if proposal >= 0:
                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        sub_graphs = self.sub_graphs_connected() #sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  #{0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])            #[[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:
            rect_set = self.rects[list(sub_graph)]     #[[228  78 238 128],[240  78 258 128]].....
            #print('rect: ', rect_set)
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


if __name__ == '__main__':
    import cv2

    rects2 = [[235, 752, 290, 332], [760, 1418, 286, 341], [555, 1098, 337, 391], [689, 965, 451, 491], [701, 953, 489, 531], [660, 860, 536, 562], [871, 905, 541, 561], [915, 951, 541, 561], [958, 992, 538, 562], [40, 96, 578, 696], [405, 535, 643, 679], [972, 1140, 660, 686], [170, 798, 716, 748], [135, 801, 747, 783], [134, 210, 784, 810], [211, 631, 779, 818], [732, 798, 784, 810], [41, 95, 807, 851], [136, 348, 816, 844], [355, 799, 813, 849], [134, 557, 845, 881], [570, 796, 850, 880], [1135, 1248, 840, 880], [136, 402, 884, 910], [468, 524, 886, 910], [522, 799, 880, 917], [135, 799, 914, 950], [1040, 1184, 918, 944], [1244, 1342, 918, 946], [35, 99, 869, 1063], [135, 555, 946, 983], [622, 799, 947, 976], [133, 753, 976, 1015], [1135, 1248, 988, 1028], [137, 175, 1021, 1041], [175, 365, 1013, 1049], [432, 798, 1014, 1044], [132, 339, 1044, 1081], [345, 799, 1045, 1082], [1036, 1180, 1064, 1092], [1239, 1346, 1065, 1094], [46, 88, 1082, 1106], [135, 537, 1078, 1116], [571, 799, 1081, 1109], [134, 800, 1113, 1146], [1342, 1398, 1120, 1146], [136, 602, 1146, 1178], [610, 798, 1148, 1174], [40, 96, 1100, 1276], [136, 798, 1180, 1213], [132, 328, 1214, 1244], [398, 799, 1209, 1249], [135, 795, 1242, 1281], [1086, 1148, 1254, 1280], [1145, 1276, 1245, 1289], [139, 799, 1279, 1316], [134, 798, 1313, 1346], [854, 966, 1320, 1352], [979, 1471, 1317, 1353], [1477, 1515, 1325, 1345], [135, 363, 1345, 1381], [853, 1516, 1345, 1389], [39, 99, 1267, 1547], [853, 1236, 1382, 1420], [1244, 1466, 1383, 1414], [854, 1040, 1420, 1452], [1037, 1519, 1416, 1457], [135, 357, 1445, 1481], [853, 1083, 1450, 1486], [1092, 1520, 1452, 1482], [855, 1517, 1483, 1521], [167, 797, 1503, 1539], [133, 197, 1535, 1571], [200, 580, 1536, 1566], [607, 796, 1538, 1571], [135, 277, 1566, 1603], [286, 798, 1570, 1600], [132, 199, 1603, 1636], [212, 244, 1606, 1630], [256, 400, 1604, 1636], [406, 576, 1604, 1630], [576, 695, 1602, 1640], [708, 798, 1604, 1630], [854, 972, 1618, 1644], [992, 1243, 1612, 1653], [1250, 1308, 1618, 1646], [1372, 1516, 1618, 1648], [131, 237, 1634, 1672], [257, 799, 1633, 1672], [855, 883, 1657, 1677], [907, 1156, 1647, 1685], [1227, 1517, 1649, 1685], [136, 238, 1670, 1698], [253, 799, 1667, 1703], [853, 1517, 1679, 1718], [135, 801, 1699, 1738], [874, 1014, 1716, 1744], [1024, 1110, 1716, 1746], [1131, 1518, 1711, 1752], [134, 799, 1731, 1775], [853, 1519, 1746, 1785], [135, 327, 1765, 1801], [335, 798, 1769, 1800], [853, 1026, 1782, 1818], [136, 518, 1802, 1832], [887, 1520, 1815, 1855], [170, 650, 1838, 1866], [654, 798, 1842, 1866], [854, 944, 1852, 1882], [967, 1519, 1851, 1887], [135, 589, 1868, 1906], [662, 798, 1870, 1902], [854, 1150, 1885, 1916], [1165, 1309, 1883, 1919], [1314, 1516, 1888, 1912], [166, 798, 1926, 1957], [853, 1429, 1915, 1953], [1452, 1516, 1918, 1946], [134, 220, 1956, 1980], [853, 1519, 1949, 1990]]
    rects = [ [ i[0], i[2], i[1], i[3] ] for i in rects2]

    #创建一个白纸
    show_image = np.zeros([2200//2, 1700//2, 3], np.uint8) + 255

    connector = BoxesConnector(rects, 2500, max_dist=15, overlap_threshold=0.3)
    new_rects = connector.connect_boxes()
    print(new_rects)

    #for rect in rects:
    #    cv2.rectangle(show_image, (rect[0]//2, rect[1]//2), (rect[2]//2, rect[3]//2), (0, 0, 255), 1)

    for rect in new_rects:
        cv2.rectangle(show_image,(rect[0]//2, rect[1]//2), (rect[2]//2, rect[3]//2),(255,0,0),1)
    cv2.imshow('res', show_image)
    cv2.waitKey(0)