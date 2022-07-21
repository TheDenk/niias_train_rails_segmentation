## Примеры обучения и инференса моделей для определения железнодорожной колеи и подвижного состава
Сайт соревнования: https://hacks-ai.ru  
  
<p>
<img src="example.png" width="1200" height="200" title="Current cutmix"/> 
</p>
  
### Установить python-venv
```python
apt install python-venv
```

Перед запуском обучения необходимо установить необходимые пакеты, находящиеся в файе requirements.txt 
### Созадать venv и установить зависимости
```python
python -m venv venv
pip install -r requirements.txt
```

### Установить Pytorch
В зависимости от машины и версии cuda необходимо ставить свою версию пайторча (опционально, requirements уже должны поставить torch).
Пример установки:
```python
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Скачать модели
```python
# python download_models.py [--out_dir]
# out_dir (необязательный) - папка, куда сохранять модели

python download_models.py --out_dir ./models
```
  
### Запустить предсказание
Для предиктов нужны модели их можно получить тремя способами:
  - выполнить скрипт download_models.py
  - скачать <a href="https://drive.google.com/drive/folders/1nA6xeQDMK_Ari8koZ8gS33p3RWAYeGuw?usp=sharing"> тут </a>.
  - обучить с помощью train.py.

Перед запуском скрипта необходимо в файле inference.py указать где лежат модели и какие у них коэфициенты. По дефолту он будет брать из папки './models'.

```python
#python inference.py [--images_dir, [--out_dir]]
# images_dir (обязательный) - путь до папки с картинками
# out_dir (необязательный) - папка, куда сохранять предсказанные маски
  
python inference.py --images_dir ./images --out_dir ./submission
```
  
### Запустить обучение
Для запуска обучения необходимо указать пути до папок с картинками и масками.  
Без указания пути сохрания, создастся папка 'models' и модели сохранятся в ней.   
Перед запуском в файле train.py раскомментировать модель, которую необходимо обучить, а остальные закомментить.

```python
# python train.py [--images_dir, --masks_dir, [--models_dir]]
# images_dir (обязательный) - путь до папки с картинками
# masks_dir (обязательный) - путь до папки с масками
# models_dir (необязательный) - папка, куда сохранять обученные модели

python train.py --images_dir ./data/images --masks_dir ./data/masks --models_dir ./models
```  

Обратная связь:  
  - Telegram: @denis_karachev
  - email: welcomedenk@gmail.com

