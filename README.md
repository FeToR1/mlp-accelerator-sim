# mlp-accelerator-sim

MLP на C++/Eigen с Python-привязкой и проектируемой моделью ускорителя
инференса полносвязных сетей.

Исходная реализация сохранена целиком в [`src/mlp/nn.cpp`](src/mlp/nn.cpp).
Python-модуль `nn` предоставляет `MLP`, `Parameters` и `SGD`; доступны
`forward`, `backward`, `predict`, `zero_grad` и `step`. Обучение и Eigen
остаются CPU-реализацией. Новая реализация ускорителя будет
развиваться рядом с исходной, которая останется доступной отдельно.

Текущая версия содержит исходную MLP, конфигурацию ускорителя и FC-команду,
воспроизводимую сборку, эксперимент на MNIST и
[описание архитектуры](docs/architecture/README.md).
Компоненты ускорителя и симулятор пока не реализованы.

## Структура

```text
CMakeLists.txt
src/mlp/nn.cpp                     исходная MLP и привязка Python
src/main.cpp                      запуск проекта
include/accelerator/config.hpp     настройки будущей модели
include/accelerator/command.hpp    описание одного FC-слоя
python/notebooks/mnist_experiment.ipynb
python/scripts/profile_memory.py   профилирование памяти CPU
docs/architecture/README.md        архитектура и схемы SVG
```

По мере реализации появятся `bindings/pybind/`,
`src/accelerator/`, `src/simulator/` и `experiments/`.

## Сборка

Нужны компилятор C++17, CMake 3.18+, Python с development headers,
Eigen 3.3+ и pybind11. Eigen устанавливается отдельно; он не включён
в репозиторий.

```sh
python -m venv .venv
```

Активируйте окружение: `source .venv/bin/activate` в Linux/macOS или
`.\.venv\Scripts\Activate.ps1` в Windows PowerShell. Затем:

```sh
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Если CMake не обнаружил Python из окружения, задайте
`-DPython_EXECUTABLE=<путь к python>`. Если Eigen скачан как набор
заголовков без установки, задайте
`-DEIGEN3_INCLUDE_DIR=<каталог, содержащий Eigen/>`.

Для MinGW на Windows используйте генератор `-G "MinGW Makefiles"`
при первоначальной конфигурации. Компилятор и `mingw32-make` должны
быть доступны в `PATH`.

Модуль собирается в `build/python/`. Скрипты и ноутбук добавляют этот
каталог в путь импорта. Для собственного Python-кода добавьте его
в `PYTHONPATH` или в `sys.path`.

## Эксперимент MNIST

После сборки запустите `python -m jupyterlab` из корня проекта и откройте
[`python/notebooks/mnist_experiment.ipynb`](python/notebooks/mnist_experiment.ipynb).
Выберите ядро из того же Python-окружения, которым собран модуль.

Ноутбук обучает сеть `784 → 128 → 10`, строит графики, оценивает
классификацию и измеряет время CPU-инференса. MNIST скачивается
в корневой `data/`, исключённый из Git. Сохранённые результаты исходного
эксперимента оставлены в ноутбуке; они не являются измерениями симулятора.

Для измерения памяти установите `memray` отдельно в поддерживаемом им
окружении Linux/macOS и выполните:

```sh
python -m pip install memray
python python/scripts/profile_memory.py
```

Скрипт использует MNIST, предварительно скачанный ноутбуком.

## Модель ускорителя

Ускоритель выполняет `Y = sigmoid(X * W^T + b)` по FIFO команд FC.
Планируются external RAM, DMA, local SRAM, tiling, microbatch,
packed weights, параметризуемый массив PE, ACC, reduction и sigmoid LUT
с линейной интерполяцией.

Симулятор запускается как C++ программа на CPU и считает работу спроектированной
архитектуры. Модельное время измеряется коммуникационными транзакциями передачи
блоков данных. Математические вычисления внутри PE выполняются обычным
C++ кодом. Базовые размеры плиток: microbatch 8, output tile 4, K tile 64.
Число PE задаётся конфигурацией перед запуском.

### Первый шаг: конфигурация

[`AcceleratorConfig`](include/accelerator/config.hpp) описывает устройство,
с которым будет работать симулятор. Все настройки задаются перед запуском
и остаются фиксированными на время одной симуляции.

| Поле | По умолчанию | Смысл |
|---|---:|---|
| `pe_count` | 4 | Число вычислительных элементов PE |
| `microbatch_size` | 8 | Максимальное число объектов, обрабатываемых одной порцией |
| `output_tile_size` | 4 | Максимальное число выходных нейронов в одной плитке |
| `k_tile_size` | 64 | Максимальное число входных признаков в одной плитке, суммарно для всех PE |

Например, у сети с 784 входными признаками `k_tile_size = 64` означает
обработку входов порциями до 64 признаков. У слоя со 128 выходами
`output_tile_size = 4` задаёт порции по четыре выхода. Microbatch ограничивает
число объектов в текущей порции, но не полный размер batch. Число PE задаёт,
между сколькими вычислителями распределяется работа этой плитки.
Само расписание обработки появится на этапе tiling.

```cpp
#include "accelerator/config.hpp"

accelerator::AcceleratorConfig config;
config.pe_count = 2;
config.validate();
```

Структура содержит четыре обычных целочисленных поля с начальными значениями.
`validate()` проверяет их положительность и сообщает об ошибке через
`std::invalid_argument`. Конфигурация не содержит веса, входные данные
или размеры сети: размеры одного слоя задаются FC-командой.

[`src/main.cpp`](src/main.cpp) пока выводит конфигурацию и одну FC-команду.
Именно `main` вызывает `config.validate()` перед использованием настроек.
Запуск после сборки:

```sh
./build/mlp_accelerator_sim
```

В Windows PowerShell запустите `./build/mlp_accelerator_sim.exe`;
при генераторе Visual Studio — `./build/Release/mlp_accelerator_sim.exe`.
Это демонстрация настроек; симулятор начнёт использовать их на следующих этапах.

### Второй шаг: FC-команда

[`FCCommand`](include/accelerator/command.hpp) описывает одну операцию
`Y = sigmoid(X * W^T + b)`:

| Поле | Смысл |
|---|---|
| `m` | Число объектов в batch |
| `n` | Число выходных нейронов |
| `k` | Число входных признаков |
| `x_addr` | Начало входов X |
| `w_addr` | Начало packed weights W |
| `bias_addr` | Начало bias |
| `y_addr` | Начало области для результатов Y |

Размеры представлены `uint32_t`, адреса — `uint64_t`. Адрес означает
смещение в байтах от начала RAM модели. Это обычные числовые поля:
сама память и обращение к ней появятся следующим этапом.

В `main` задан слой `784 → 128` для восьми объектов. Адреса пока условные:
области RAM по ним ещё не выделяются. Команда только создаётся и выводится.
Размеры `m/n/k` описывают весь слой, а размеры плиток из конфигурации —
порции, которыми этот слой будет обрабатываться.

Разработка идёт отдельными этапами и коммитами: config; FC command;
RAM и packing; счётчики транзакций; DMA/SRAM;
PE; ACC/reduction; tiling; LUT; FIFO; интеграция; profiling; experiments.
