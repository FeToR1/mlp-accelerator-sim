# mlp-accelerator-sim

MLP на C++/Eigen с Python-привязкой и проектируемой моделью ускорителя
инференса полносвязных сетей.

Исходная реализация сохранена целиком в [`src/mlp/nn.cpp`](src/mlp/nn.cpp).
Python-модуль `nn` предоставляет `MLP`, `Parameters` и `SGD`; доступны
`forward`, `backward`, `predict`, `zero_grad` и `step`. Обучение и Eigen
остаются CPU/reference implementation. Новая реализация ускорителя будет
развиваться рядом с исходной, которая останется доступной отдельно.

Текущая версия содержит исходную MLP, отдельный C++ inference API,
воспроизводимую сборку, эксперимент на MNIST и
[описание архитектуры](docs/architecture/README.md).
Компоненты ускорителя и симулятор пока не реализованы.

## Структура

```text
CMakeLists.txt
src/mlp/nn.cpp                     исходная MLP и привязка Python
include/mlp/reference.hpp          публичный C++ inference API
src/mlp/reference.cpp              Eigen FC и последовательность слоёв
tests/                            численные проверки и сравнение с nn
python/notebooks/mnist_experiment.ipynb
python/scripts/check_reference.py  проверка собранного модуля
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
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake -E chdir build ctest -C Release --output-on-failure
python python/scripts/check_reference.py
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

## C++ inference API

Библиотека `mlp_reference` зависит от Eigen и принимает параметры явно.
`mlp::Layer` хранит веса `[K, N]` в row-major порядке и bias `[N]`.
`mlp::dense` вычисляет один FC-слой; `mlp::predict` последовательно выполняет
непустой список слоёв. Параметры и входы не изменяются, градиенты и обучающие
активации не хранятся. Несовместимые или пустые размеры вызывают
`std::invalid_argument`. Sigmoid сохраняет формулу исходного Eigen backend.

```cpp
#include "mlp/reference.hpp"

mlp::Matrix x = mlp::Matrix::Ones(2, 3);
mlp::Layer layer{mlp::Matrix::Constant(3, 4, 0.1f),
                 mlp::Vector::Zero(4)};
mlp::Matrix y = mlp::dense(x, layer);       // [2, 4]
mlp::Matrix result = mlp::predict(x, {layer});
```

CMake-потребитель подключает `target_link_libraries(app PRIVATE mlp_reference)`.
Исходный Python-модуль `nn` остаётся отдельным. CTest проверяет FC против
скалярных dot products, несколько слоёв, sigmoid, ошибки размеров и совпадение
с `nn.MLP.predict` на трёх топологиях с одинаковыми входами и весами.

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

Модельное время измеряется коммуникационными транзакциями передачи
блоков данных. Математические вычисления внутри PE выполняются обычным
C++ кодом. Базовые размеры плиток: microbatch 8, output tile 4, K tile 64.
Число PE задаётся конфигурацией перед запуском.

Разработка идёт отдельными этапами и коммитами: reference API;
config и FC command; RAM и packing; счётчики транзакций; DMA/SRAM;
PE; ACC/reduction; tiling; LUT; FIFO; интеграция; profiling; experiments.
