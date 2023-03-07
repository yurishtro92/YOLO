import cv2
import numpy as np


def apply_yolo_object_detection(image_to_process):
    """Распознавание и определение координат объектов на изображении
        :param image_to_process: исходное изображение
        :return: изображение с размеченными объектами и подписями к ним"""
    # try:
    #     pass
    # except AttributeError:
    height, width, depth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    # try:
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0
    # except IndexError:
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_х = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_х - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Выборка
    chosen_boxes = cv2.dnn.NМSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index[0]
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # Для отладки рисуем объекты, входящие в искомые классы
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process, class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)
    return final_image


def draw_object_bounding_box(image_to_process, index, box):
    """Рисование границ объекта с подписями
    :param image_to_process: исходное изображение
    :param index: индекс определенного с помощью YOLO класса объекта
    :param bох: координаты области вокруг объекта
    :return: изображение с отмеченными объектами
    """

    х, у, w, h = box
    start = (х, у)
    end = (х + w, у + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (х, у - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)
    return final_image


def start_image_object_detection():
    """Анализ изображения"""
    try:
        # Применение методов распознавания объектов на изображении от YOLO
        image = cv2.imread("/truck_captcha.png")
        image = apply_yolo_object_detection(image)
        # Вывод обработанного изображения
        cv2.imshow("Image", image)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # Загрузка весов YOLO из файлов и настройка сети
    try:
        net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
        layer_names = net.getLayerNames()
    except IndexError:
        out_layers_indexes = net.getUnconnectedOutLayers()
        out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]
        # Загрузка из файла классов объектов, которые умеет обнаруживать YOLO
    with open("coco_names.txt") as file:
        classes = file.read().split("\n")
    # Определение классов, которые будут приоритетными для поиска на изображении
    # Названия находятся в файле coco.names.txt
    # В данном случае определяется грузовик для прохождения CAPTCHA
    classes_to_look_for = ["truck"]

    start_image_object_detection()
