# image_classification
Это веб-приложение, предназначенная для классификации изображений из категорий "лес", "улица", "море", "ледник".
В сонове используется классификатор `RandomForestClassifier` из библиотеки `sklearn`. 

### Датасет

Этот набор данных состоит из изображений, охватывающих четыре категории природных сцен, предназначенных для задач классификации изображений.
Он включает следующие категории:

1. **Лес**: Фотографии лесов и деревьев.
2. **Улица**: Снимки улиц и городских сцен.
3. **Море**: Фотографии морей и океанов.
4. **Ледник**: Снимки ледников и заснеженных пейзажей.

[Dataset link](https://www.kaggle.com/datasets/rahmasleam/intel-image-dataset)

### Использование

Инициализация:

```
make init
```

Запуск приложения:

```
make app
```

> поддерживаются `.png`, `.jpg`, `.jpeg` файлы

### Архитектура

**Основные компоненты приложения**:
   - **Объектная модель фич**: Каждый тип фичи реализуется как отдельный класс, наследующий от абстрактного класса `IFeature`. Это позволяет легко добавлять новые фичи в будущем и использовать полиморфизм.
   - **Фичи (Features)**: Все фичи, такие как `HorizontalEdgeCount`, `VerticalEdgeCount`, `GreenPixelPercentage` и другие, вычисляют специфические характеристики изображений, которые затем используются для классификации.
   - **Классификатор**: Использует рассчитанные фичи для обучения модели классификации и классификации новых изображений.

**Этапы работы приложения**:
   - **Загрузка изображения**: Изображение загружается в приложение как объект `np.ndarray` (массив данных изображения).
   - **Вычисление фич**: Для каждого изображения происходит расчет фич. Каждый класс фичи реализует метод `calculate`, который обрабатывает изображение и возвращает числовое значение (например, процент зеленых пикселей для фичи `GreenPixelPercentage`).
   - **Предобработка изображения**: Если изображение цветное, оно преобразуется в оттенки серого перед применением большинства фич. В случае фич с использованием цветовых каналов (например, `GreenPixelPercentage`, `BluePixelPercentage` и т.д.)
используется соответствующий канал.
   - **Обучение классификатора**: После извлечения фич из набора изображений, они передаются в классификатор (RandomForest) для тренировки модели.
   - **Классификация**: После тренировки классификатор может использовать вычисленные фичи для определения категории нового изображения (например, будет ли оно изображением леса, моря и т.д.).


### Тестирование

Тестирование проходило по двум сценариям:

1. Тестирование обученного классификатора (использовался тестовый датасет, который не участвовал в обучении)
    - Для ознакомления с поддробным результатом обратитесь [сюда](https://github.com/vladyoslav/image-classifier/blob/main/notebooks/classifier.ipynb)
2. Unit тестирование
<details>
  <summary>Coverage Unit Tests</summary>

| Файл                                | Строки | Пропущено | Покрытие (%) |
|-------------------------------------|--------|-----------|--------------|
| `src/__init__.py`                   | 2      | 0         | 100%         |
| `src/classifier/__init__.py`        | 2      | 0         | 100%         |
| `src/classifier/classifier.py`      | 17     | 0         | 100%         |
| `src/features/__init__.py`          | 2      | 0         | 100%         |
| `src/features/features.py`          | 73     | 4         | 95%          |
| `src/utils/__init__.py`             | 2      | 0         | 100%         |
| `src/utils/utils.py`                | 16     | 0         | 100%         |
| **Итого**                           | 114    | 4         | 96%          |

</details>
3. Ручное тестирование веб-приложения на следующих изображениях:

  - [лес1](https://rewildingeurope.com/wp-content/uploads/2021/11/RS1765_SBA_2011-07-30_070027-low-e1646985237178.jpg)
  - [лес2](https://www.massaudubon.org/var/site/storage/images/9/3/6/2/1602639-1-eng-US/2fe6f73efaea-RE_KForesto-3461-1920x1280.jpg)
  - [лес3](https://www.metroparks.net/wp-content/uploads/2017/06/1080p_HBK_autumn-morning_GI.jpg)
  - [лес4](https://www.nationalforests.org/assets/header-images/Siuslaw-NF_Drift-Creek_Sam-Beebe-1200x800-qual30.jpg)
  - [лес5](https://www.charlottenewsvt.org/wp-content/uploads/2018/09/Forest_web.jpg)
  
  - [ледник1](https://nsidc.org/sites/default/files/images/Exploring%20the%20Perito%20Moreno%20Glacier_Bruce%20Raup_2019-12-19-33_6329.jpg)
  - [ледник2](https://cdn.mos.cms.futurecdn.net/6VYFeFUWtYEomkuoZQc4hB.jpg)
  - [ледник3](https://www.gns.cri.nz/assets/Uploads/Heroes/glacier-hero__FillWzEyMDAsNjAwXQ.jpg)
  - [ледник4](https://www.whoi.edu/wp-content/uploads/2022/09/glacier-amy-kukulya.jpg)
  
  - [море1](https://media-cdn.tripadvisor.com/media/photo-s/1a/5c/b9/f3/sea-view-beach-karachi.jpg)
  - [море2](https://wallpapershome.com/images/pages/pic_v/25805.jpg)
  - [море3](https://cdn.pixabay.com/video/2023/07/12/171277-845168284_tiny.jpg)
