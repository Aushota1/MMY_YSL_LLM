# Быстрое создание датасета для обучения Embeddings

## 🚀 Самый простой способ

```bash
# Использует war_and_peace.ru.txt (если существует)
python create_embedding_dataset.py
```

Результат: файл `embedding_dataset.txt` готов к использованию!

## 📝 Более полный датасет

```bash
python create_embedding_dataset.py \
    --files war_and_peace.ru.txt \
    --synthetic \
    --add-samples \
    --output large_dataset.txt
```

Это создаст:
- ✅ Датасет из war_and_peace.ru.txt
- ✅ Увеличенный в 2 раза синтетическими текстами
- ✅ С добавлением примеров разных типов текстов
- ✅ Сохраненный в large_dataset.txt

## 🎯 Использование для обучения

```bash
# 1. Создать датасет
python create_embedding_dataset.py --files war_and_peace.ru.txt

# 2. Обучить embeddings
python EMBEDDING_LAYER/auto_evaluation_system.py \
    --data embedding_dataset.txt \
    --epochs 10 \
    --batch-size 32
```

## 📊 Что делает скрипт?

1. **Загружает** тексты из файлов
2. **Очищает** текст (удаляет лишнее, нормализует)
3. **Разбивает** на предложения (умно, учитывая сокращения)
4. **Фильтрует** по длине (удаляет слишком короткие/длинные)
5. **Удаляет** дубликаты
6. **Сохраняет** в удобном формате (одна строка = одно предложение)

## 💡 Полезные опции

```bash
# Только предложения 20-200 слов
--min-length 20 --max-length 200

# Увеличить датасет в 5 раз
--synthetic --synthetic-multiplier 5

# Загрузить из нескольких файлов
--files file1.txt file2.txt file3.txt

# Загрузить из директории
--dirs ./my_texts
```

## 📈 Результат

После выполнения вы увидите статистику:
- Сколько предложений создано
- Сколько уникальных слов
- Размер файла
- И многое другое!

---

**Подробная документация**: `DATASET_BUILDER_README.md`

