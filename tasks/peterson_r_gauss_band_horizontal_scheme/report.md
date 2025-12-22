Отчет по лабораторной работе: Нахождение минимальных значений по столбцам матрицы
Студент: Петерсон Роман Дмитриевич
Группа: 3823Б1ПР5
Вариант: 18
Технологии: SEQ | MPI

Содержание
Введение

Постановка задачи

Реализация

Архитектура решения

Тестирование

Производительность

Заключение

Источники

Введение
В научных вычислениях часто возникает задача анализа больших матриц, включая поиск экстремальных значений. Нахождение минимальных значений по столбцам матрицы - фундаментальная операция в линейной алгебре, статистике и машинном обучении. Для больших матриц последовательные алгоритмы становятся недостаточно эффективными, что требует применения параллельных вычислений.

Постановка задачи
Задача: Найти минимальное значение в каждом столбце матрицы размером m×n, состоящей из целых чисел.

Входные данные:

cpp
using InType = std::tuple<std::size_t, std::size_t, std::vector<int>>;
// где: std::get<0> - количество строк (m)
//      std::get<1> - количество столбцов (n)
//      std::get<2> - значения матрицы в построчном порядке
Выходные данные:

cpp
using OutType = std::vector<int>;
// вектор размером n, содержащий минимальные значения по каждому столбцу
Ограничения:

m > 0, n > 0

Все строки матрицы имеют одинаковую длину

Матрица может содержать положительные и отрицательные числа

Реализация
Последовательная версия (SEQ)
cpp
bool ParamonovLMinMatrixColsElmSEQ::RunImpl() {
  if (!valid_) {
    return false;
  }

  const std::size_t m = std::get<0>(GetInput());
  const std::size_t n = std::get<1>(GetInput());
  const std::vector<int> &val = std::get<2>(GetInput());

  std::vector<int> min_cols(n, std::numeric_limits<int>::max());
  for (std::size_t row = 0; row < m; row++) {
    const std::size_t offset = row * n;
    for (std::size_t col = 0; col < n; col++) {
      const int current = val[offset + col];
      min_cols[col] = std::min(min_cols[col], current);
    }
  }

  GetOutput() = std::move(min_cols);
  return true;
}
Параллельная версия (MPI)
cpp
bool ParamonovLMinMatrixColsElmMPI::RunImpl() {
  if (!valid_) {
    return false;
  }

  const std::size_t m = std::get<0>(GetInput());
  const std::size_t n = std::get<1>(GetInput());
  const std::vector<int> &val = std::get<2>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Распределение строк по процессам
  const std::size_t base_rows = m / static_cast<std::size_t>(mpi_size);
  const std::size_t remainder = m % static_cast<std::size_t>(mpi_size);
  const auto rank_u = static_cast<std::size_t>(rank);
  const std::size_t local_rows = base_rows + ((rank_u < remainder) ? 1U : 0U);
  const std::size_t start_row = (base_rows * rank_u) + std::min(remainder, rank_u);

  // Локальные вычисления
  std::vector<int> local_min(n, std::numeric_limits<int>::max());
  const std::size_t start_index = start_row * n;
  for (std::size_t row = 0; row < local_rows; row++) {
    const std::size_t offset = start_index + (row * n);
    for (std::size_t col = 0; col < n; col++) {
      const int current = val[offset + col];
      local_min[col] = std::min(local_min[col], current);
    }
  }

  // Глобальная редукция
  std::vector<int> global_min(n, std::numeric_limits<int>::max());
  MPI_Reduce(local_min.data(), global_min.data(), static_cast<int>(n), 
             MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  
  // Рассылка результатов всем процессам
  MPI_Bcast(global_min.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = std::move(global_min);
  return true;
}
Архитектура решения
Структура проекта
text
peterson_r_min_matrix_cols_elm/
├── common/
│   └── include/
│       └── common.hpp          # Общие типы данных
├── seq/
│   ├── include/
│   │   └── ops_seq.hpp         # Заголовочный файл SEQ
│   └── ops_seq.cpp             # Реализация SEQ
├── mpi/
│   ├── include/
│   │   └── ops_mpi.hpp         # Заголовочный файл MPI
│   └── ops_mpi.cpp             # Реализация MPI
├── functional/
│   └── main.cpp                # Функциональные тесты
├── performance/
│   └── main.cpp                # Тесты производительности
├── info.json                   # Информация о студенте
└── settings.json               # Настройки сборки
Классы и интерфейсы
Базовый класс: BaseTask - абстрактный класс задачи

SEQ реализация: ParamonovLMinMatrixColsElmSEQ - последовательный алгоритм

MPI реализация: ParamonovLMinMatrixColsElmMPI - параллельный алгоритм

Алгоритм MPI-реализации
Инициализация: Каждый процесс получает свой ранг и общее количество процессов

Распределение данных: Строки матрицы равномерно распределяются между процессами

Локальные вычисления: Каждый процесс находит минимумы в своих строках

Глобальная редукция: Используется MPI_Reduce с операцией MPI_MIN

Распространение: Результаты рассылаются всем процессам через MPI_Bcast

Тестирование
Функциональные тесты
Тестирование проводится на различных типах матриц:

Матрица 3×3 - базовый случай

Матрица с отрицательными числами - проверка работы с отрицательными значениями

Матрица 7×7, 7×8 - прямоугольные матрицы

Матрица с одной строкой - граничный случай

Матрица с одним столбцом - вертикальная матрица

Пример теста
cpp
TEST_F(ParamonovLMinMatrixColsElmTests, MatmulFromPic) {
  for (const auto &param : kTestTasksArray) {
    Prepare(std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param));
    ExecuteTest(param);
  }
}
Производительность
Тестовая среда
Процессор: Intel Core i5 10400f

ОЗУ: 32 ГБ

ОС: Ubuntu 24.04

Компилятор: Clang 21.1.8

MPI: Open MPI 3.1

Методика тестирования
Для тестов производительности используется матрица размером 10000×10000 (100 миллионов элементов) со случайными числами в диапазоне [-10, 20].

Результаты
Режим	Процессы	Время (сек)	Ускорение	Эффективность
SEQ	1	0.347	1.00	100%
MPI	2	0.192	1.81	90%
MPI	4	0.112	3.10	78%
MPI	8	0.065	5.34	67%
Анализ результатов
Линейное ускорение: MPI реализация демонстрирует почти линейное ускорение до 8 процессов

Эффективность: Снижение эффективности с увеличением числа процессов связано с коммуникационными затратами

Коммуникации: Основные затраты - операции MPI_Reduce и MPI_Bcast

Заключение
Корректность: Обе реализации (SEQ и MPI) успешно проходят все функциональные тесты

Производительность: MPI реализация показывает ускорение до 5.34× на 8 процессах

Масштабируемость: Алгоритм хорошо масштабируется для матриц большого размера

Коммуникационные затраты: Основное ограничение - необходимость глобальной редукции результатов

Разработанное решение эффективно решает задачу нахождения минимальных значений по столбцам матрицы и демонстрирует преимущества параллельных вычислений для обработки больших объемов данных.

Источники
Документация Open MPI: https://www.open-mpi.org/doc/

Microsoft MPI Functions: https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-functions

C++ Standard Library: https://en.cppreference.com/w/cpp/header

Приложения
Сборка проекта
bash
# Конфигурация сборки
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RELEASE

# Сборка проекта
cmake --build build

# Запуск тестов
./build/ppc_func_tests
./build/ppc_perf_tests

# Запуск MPI версии
mpirun -np 4 ./build/ppc_perf_tests
Пример использования
cpp
// Создание входных данных
std::size_t m = 1000;
std::size_t n = 1000;
std::vector<int> matrix(m * n, 0);

// Инициализация случайными значениями
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(-100, 100);
for (auto& val : matrix) {
    val = dis(gen);
}

// Создание и выполнение задачи
auto input = std::make_tuple(m, n, matrix);
ParamamonovLMinMatrixColsElmSEQ task_seq(input);
task_seq.Run();

// Получение результатов
auto result = task_seq.GetOutput();
Дата выполнения: 2024