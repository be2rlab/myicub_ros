Функция для загрузки предобученного объекта: load_state(path='detector.pckl)

Датасет необходимо загрузить в папку: \
dataset\
-images\
-labels

Класс Detector:\
  Для обучения метод train(train_file, class_names, valid_file=None)\
  аргументы: train_file - файл со списком файлов датасета (с абсолютными/относительными путями)\
             class_names - один или список названий классов\
             valid_file - опционально, файл с указанием путей для тестирования в процессе обучения
             
  Для детекции метод detect(image)\
  Аргументы: image - изображение в виде массива точек\
  Возвращает: новое изображение с рамками и координаты рамок с указанием классов
  
  Для поиска определенного объекта на сцене метод find_object(object_name, image)\
  Аргументы: название класса, изображение\
  Возвращает: координаты рамок в формате: [x1, y1, x2, y2, confidence, class_index]\
        Если неизвестный объект - "unknown object"\
        Если объект не найден - "not found"
        
  Для сохранения состояния объекта метод save_state(path='detector.pckl') 

Датасет https://drive.google.com/file/d/1ycpMPbZnskCVWeyZrMf9CoLGFJ9xrRNJ/view?usp=sharing

