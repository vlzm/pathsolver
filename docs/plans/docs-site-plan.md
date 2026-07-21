# План: сайт документации Pathsolver на базе шаблона TransTTE

> Цель: переделать скопированный в `docs/site/` Hugo-сайт (тема **hugo-book**,
> документация TransTTE) в документацию проекта Pathsolver — «Zero-Knowledge ML for
> Finding Short Paths on Large Cayley Graphs» (NeurIPS 2025 Spotlight,
> [arXiv:2502.13266](https://www.arxiv.org/pdf/2502.13266)).

## Принятые решения

- **Язык документации — английский** (проект публичный, NeurIPS).
- **Деплой — GitHub Pages** репозитория `https://github.com/vlzm/pathsolver`
  → `baseURL = 'https://vlzm.github.io/pathsolver/'`.
- Структура разделов — по образцу TransTTE: Research → Architecture → Usage → Reference.

## Как устроен шаблон (справка)

- Hugo + тема **hugo-book** (вендорена в `docs/site/themes/hugo-book`, внешних
  зависимостей нет). Локальный запуск: `hugo server` из `docs/site/`.
- **Меню строится из дерева** `content/docs/<раздел>/<страница>.md`. Порядок —
  `weight:` во front matter; сворачиваемый раздел — `bookCollapseSection: true`
  в `_index.md` раздела.
- **Формулы**: страница объявляет `math: true` → кастомный партиал
  `layouts/partials/docs/inject/head.html` подключает KaTeX. Это единственная
  кастомизация темы — **сохранить её**. Инлайн-математика `\( ... \)`, блочная —
  ` ```katex `-блоки.
- **Диаграммы**: `mermaid: true` во front matter + ` ```mermaid `-блоки.
- Конфиг — единственный `hugo.toml` (baseURL, title, BookRepo/BookEditPath для
  ссылки «Edit this page», BookSearch).
- `public/` и `resources/` — артефакты сборки старого сайта, `static/images/` —
  картинки TransTTE. Всё это подлежит удалению/замене.

## Этап 1 — перенастройка каркаса

1. Удалить артефакты сборки: `docs/site/public/`, `docs/site/resources/`,
   `docs/site/.hugo_build.lock`; добавить их в `.gitignore`.
2. Удалить контент TransTTE: всё в `docs/site/content/`, картинки
   `docs/site/static/images/transtte_*.png`.
3. `hugo.toml`:
   - `baseURL = 'https://vlzm.github.io/pathsolver/'`
   - `title = 'Pathsolver'`
   - `BookRepo = 'https://github.com/vlzm/pathsolver'`
   - `BookEditPath = 'edit/main/docs/site/content'`
4. Скопировать `assets/fig.png` (схема пайплайна из README) в
   `docs/site/static/images/`.

## Этап 2 — структура контента

```
content/
  _index.md                      — лендинг: NeurIPS'25 Spotlight, arXiv:2502.13266,
                                   fig.png, таблица результатов из README, quick links
  docs/
    _index.md                    — обзор документации, ссылки на разделы
    introduction/_index.md       — что это: zero-knowledge поиск коротких путей на
                                   графах Кэли; одна архитектура/гиперпараметры — все пазлы
    research/
      _index.md
      problem-formulation.md     — графы Кэли, пермутационные пазлы, почему классический
                                   поиск не масштабируется (math: true)
      method.md                  — оценка диффузионного расстояния нейросетью +
                                   батчевый GPU beam search (math: true)
      results.md                 — таблицы: 3x3x3 / 4x4x4 / Pancake 55 / Klotski 6x6;
                                   сравнение с DeepCubeA, EfficientCube, Santa-2023;
                                   98% оптимальности на 3x3x3 QTM; 20x speedup
    architecture/
      _index.md                  — обзор + mermaid-диаграмма пайплайна
      pipeline.md                — генерация данных → обучение → beam search → ONNX → UI
      pilgrim.md                 — пакет pilgrim/: model.py, trainer.py, searcher.py,
                                   dqn.py, utils.py (~1000 строк)
      data-contract.md           — конвенции данных: generators/pXXX.json,
                                   targets/*.pt, datasets/pXXX-tXXX-{rnd,santa}.pt
                                   (аналог "weight contract" в шаблоне)
    usage/
      _index.md
      testing.md                 — test.py с предобученными весами (weights/),
                                   команды для 3x3x3/4x4x4/5x5x5 из README
      training.md                — train.py, гиперпараметры (hd1/hd2/nrd/K_max,
                                   batch_size), model_id = int(time.time())
      multiagent.md              — traintest-multiagent.sh, per-agent и ансамблевая
                                   статистика, read-test-logs-multiagent.py
      new-puzzles.md             — добавление своего пазла: generators + targets + datasets
      onnx-and-ui.md             — export_onnx.py, onnx/, веб-демо ui/
    reference/
      _index.md
      cli.md                     — все флаги train.py / test.py (брать из argparse
                                   в коде, НЕ из README — там неполный список)
      groups.md                  — таблица Group ID / пазл / Kmax (из README)
      logs.md                    — формат logs/test_*.json, read-test-logs.py
      glossary.md                — QTM/UQTM, beam width B, scramble, K_max,
                                   diffusion distance, Cayley graph…
```

Опционально (позже): раздел **LRX + MDQN** по мере выполнения
`docs/plans/lrx-mdqn-integration-plan.md` (этап 1 — MDQN в `pilgrim` — уже готов,
страницу про `Trainer.run_dqn` и CLI-флаги можно писать сразу).

## Этап 3 — наполнение (порядок по трудоёмкости)

Источники — всё уже в репозитории:

| Раздел | Источник |
|---|---|
| usage/*, reference/groups | README.md (переносится почти дословно, разбить на страницы) |
| reference/cli | argparse в `train.py`, `test.py` |
| reference/logs | `read-test-logs.py`, `read-test-logs-multiagent.py`, файлы в `logs/` |
| architecture/* | код `pilgrim/`, `train.py`, `test.py`, `export_onnx.py`, `ui/` |
| research/* | `docs/references/2502.13266v1.pdf` (основная статья); для LRX — `2502.18663v1.pdf` |
| лендинг | шапка README + `assets/fig.png` |

Рекомендуемый порядок написания: **usage → reference → architecture → research**
(от быстрого к трудоёмкому; research требует чтения статьи).

## Этап 4 — сборка и деплой

1. Локальная проверка: `cd docs/site && hugo server` — проверить меню, поиск,
   KaTeX (`math: true`), mermaid, относительные ссылки (`relref`).
2. GitHub Actions workflow (в скопированном сайте его нет — остался в старом
   репозитории): стандартная пара `actions/configure-pages` + сборка Hugo +
   `actions/deploy-pages`, триггер на push в `main` по пути `docs/site/**`.
   В настройках репозитория включить Pages (Source: GitHub Actions).
3. Проверить готовый сайт по `https://vlzm.github.io/pathsolver/`.

## Definition of done

- [ ] Никаких упоминаний TransTTE в `content/`, `hugo.toml`, `static/`
- [ ] Все страницы из дерева этапа 2 существуют и заполнены (не заглушки)
- [ ] `hugo` собирается без ошибок и warning'ов о битых relref
- [ ] Формулы (KaTeX) и mermaid-диаграммы рендерятся
- [ ] Сайт опубликован на GitHub Pages, ссылки «Edit this page» ведут в репозиторий
