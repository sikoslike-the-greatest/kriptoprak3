# Практическая работа 5 — Встраивание ЦВЗ в частотную область изображений

Программная реализация **гибридного алгоритма встраивания цифровых водяных
знаков (ЦВЗ) на основе ДКП и ДВП** (Abdulrahman & Ozturk, 2019) для дисциплины
«Основы криптографии и стеганографии».

## Метод

1. К полутоновому изображению-контейнеру применяется полнокадровое ДКП.
2. К массиву коэффициентов ДКП применяется один уровень ДВП Хаара (поддиапазоны
   `LL`, `HL`, `LH`, `HH`).
3. Бинарный ЦВЗ перемешивается преобразованием Арнольда, к нему применяется ДКП,
   результат делится на 4 блока.
4. Блоки аддитивно встраиваются в левые верхние углы поддиапазонов:
   `ID' = ID + alpha * W`.
5. Обратное ДВП и обратное ДКП дают изображение со встроенным ЦВЗ.

Извлечение **неслепое** — требует исходного контейнера.

## Структура

```
dwtdct/
  watermark.py     # ядро: ДВП Хаара, Арнольд, embed/extract
  metrics.py       # PSNR, MSE, RMSE, SSIM, BER, NCC
  cli.py           # CLI: embed / extract
  experiments.py   # вычислительные эксперименты -> images/
images/            # рисунки и таблицы (генерируются)
report5.tex        # отчёт (xelatex)
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Использование

```bash
# встраивание
python -m dwtdct embed   --cover cover.png --watermark wm.png \
                         --stego stego.png --params params.json --alpha 20 --iters 10

# извлечение (неслепое: нужен исходный контейнер)
python -m dwtdct extract --stego stego.png --cover cover.png \
                         --params params.json --output recovered.png --watermark wm.png
```

## Эксперименты и отчёт

```bash
python -m dwtdct.experiments      # генерирует рисунки и таблицы в images/
xelatex report5.tex && xelatex report5.tex
```
