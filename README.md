# Лабораторная работа №3

[Архитектура](docs/architecture/README.md) · [Отчёт](docs/reports/profiling.md)

Нужны C++17, CMake, SystemC 3.0+, Eigen 3.3+ и MinGW в PATH.
Для SystemC указан локальный каталог установки `.local/systemc`.

```powershell
$env:PATH = "$env:LOCALAPPDATA\mlp_cpp\mingw64\bin;$env:PATH"
cmake -S . -B build -G "MinGW Makefiles" "-DCMAKE_PREFIX_PATH=.local/systemc" -DBUILD_PYTHON_MODULE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2
.\build\mlp_accelerator_sim.exe
.\build\mlp_accelerator_sim.exe --mlp 784,128,10 4 8
python experiments/run.py
```

`main` сверяет результат с CPU с допуском `1e-4`.
CSV новых замеров сохраняются в `experiments/results/`.

В Python после обучения сети с sigmoid после каждого Linear:

```python
torch.save(model.state_dict(), "data/model.pt")
torch.save(images.flatten(1), "data/input.pt")
```

```powershell
python python/scripts/export_pytorch.py data/model.pt data/model.txt --input data/input.pt
.\build\mlp_accelerator_sim.exe --weights data/model.txt 4 --input data/model.input.txt
```

Без `--input` используются случайные входы; batch задаётся после числа PE.

Замеры при batch=8, PE=4, плитках 8/4/64.
Память включает RAM, SRAM, ACC, bias, LUT и FIFO.
Загрузка PE — доля полезных MAC в ёмкости обработанных плиток.

| Сеть | Время модели | Память, байт | PE0, % | PE1, % | PE2, % | PE3, % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 784→128→10 | 12694 | 495804 | 94.08 | 94.08 | 94.08 | 94.08 |
| 784→256→128→10 | 29527 | 1053884 | 94.93 | 94.93 | 94.93 | 94.93 |
| 784→512→256→128→10 | 70360 | 2432188 | 95.80 | 95.80 | 95.80 | 95.80 |
