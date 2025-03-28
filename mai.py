import numpy as np
from random import randint, sample

def create_random_matrix(N, M, t1, t2):
    matrix = np.random.randint(t1, t2 + 1, size=(M, N)).astype(float)  # Ensure it's float for inf
    for row in matrix:
        if randint(0, 1) == 1:
            k = randint(1, N - 1)
            indices_to_change = sample(range(N), k)
            for idx in indices_to_change:
                row[idx] = np.inf
    return matrix

def sorting(number, matrix):
    """Сортировка матрицы."""
    if number == 1:
        inf_counts = np.sum(matrix == np.inf, axis=1)
        sorted_indices = np.argsort(inf_counts)[::-1]
        return matrix[sorted_indices]
    elif number == 2:
        def sort_key(row):
            inf_count = np.sum(row == np.inf)
            row_without_inf = np.where(row == np.inf, 0, row)
            row_sum = np.sum(row_without_inf)
            return (inf_count, row_sum)

        sorted_indices = sorted(range(matrix.shape[0]), key=lambda i: sort_key(matrix[i]), reverse=True)
        return matrix[sorted_indices]
    elif number == 3:
        return matrix
    else:
        print("Неверный номер метода сортировки.")
        return matrix

def ALG_minimax(matrices, file):

    print("\nПрименение алгоритма Minimax...", file=file)
    print("\nПрименение алгоритма Minimax...")
    results = []
    for matrix_index, matrix in enumerate(matrices):
        print(f"\nМатрица {matrix_index}:", file=file)
        print(f"\nМатрица {matrix_index}:")
        M, N = matrix.shape
        sums = np.zeros(N)  # Загрузка каждого исполнителя
        assignment = [-1] * M

        for i in range(M):
            best_j = -1
            min_load = np.inf

            for j in range(N):
                if matrix[i, j] != np.inf:
                    if sums[j] < min_load:
                        min_load = sums[j]
                        best_j = j

            if best_j != -1:
                sums[best_j] += matrix[i, best_j]
                assignment[i] = best_j
                print(f"  Задача {i}: назначена исполнителю {best_j}, вя: {matrix[i, best_j]}", file=file)
            else:
                best_j = np.argmin(sums)
                sums[best_j] += matrix[i, best_j]
                assignment[i] = best_j
                print(f"  Предупреждение: Задача {i} назначена исполнителю {best_j}, хотя он не подходит (все inf).", file=file)
        print("  загрузка задач:", sums, np.max(sums))
        print("  Распределение задач:", assignment, file=file)
        print("  Суммарная загрузка исполнителей:", sums, file=file)
        print("  Максимальная загрузка:", np.max(sums), file=file)
        results.append({"matrix_index": matrix_index, "assignment": assignment, "sums": sums, "max_load": np.max(sums), "criterion": "minimax"})
    return results


def ALG_quadratic(matrices, file):
    print("\nПрименение алгоритма Плотникова (степень: 2)...", file=file)
    print("\nПрименение алгоритма Плотникова (степень: 2)..." )
    results = []
    for matrix_index, matrix in enumerate(matrices):
        print(f"\nМатрица {matrix_index}:", file=file)
        print(f"\nМатрица {matrix_index}:")
        M, N = matrix.shape
        sums = np.zeros(N)
        assignment = [-1] * M
        all_inf = False

        for i in range(M):
            best_j = -1
            min_cost = np.inf
            all_inf = True  # Флаг, показывающий, что все процессоры недоступны

            print(f"  \nЗадача {i}:", file=file)
            for j in range(N):
                if matrix[i, j] != np.inf:
                    all_inf = False
                    #  Вычисляем общую "стоимость" системы, если назначить задачу процессору j
                    total_cost = 0
                    cost_components = []
                    for k in range(N):
                        if k == j:
                            processor_cost = (sums[k] + matrix[i, j])**2
                            cost_components.append(f"({sums[k]} + {matrix[i, j]})^2")
                        else:
                            processor_cost = sums[k]**2
                            cost_components.append(f"({sums[k]})^2")
                        total_cost += processor_cost
                    cost_string = " + ".join(cost_components) + f" = {total_cost}"
                    print(f"    изадачу {i} процессору {j}:", file=file)
                    print(f"      costs = {cost_string}", file=file)

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_j = j


            if not all_inf:
                sums[best_j] += matrix[i, best_j]
                assignment[i] = best_j
                print(f"  -> Выбираем процессор {best_j}: processors[{best_j}] += {matrix[i, best_j]}", file=file)
            else:  #  Если все процессоры inf, выбираем процессор с минимальной cost
                print("    Все процессоры недоступны. Выбираем лучший по стоимости...", file=file)
                best_j = -1
                min_cost = np.inf
                for j in range(N):  # Снова перебираем все процессоры
                    total_cost = 0
                    cost_components = []
                    for k in range(N):
                        if k == j:
                            processor_cost = (sums[k] + matrix[i, j])**2  # matrix[i, j] == inf
                            cost_components.append(f"({sums[k]} + {matrix[i, j]})^2")
                        else:
                            processor_cost = sums[k]**2
                            cost_components.append(f"({sums[k]})^2")
                        total_cost += processor_cost
                    cost_string = " + ".join(cost_components) + f" = {total_cost}"
                    print(f"    Процессор {j}: cost = {cost_string}", file=file)

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_j = j
                sums[best_j] += matrix[i, best_j]
                assignment[i] = best_j
                print(f"    Выбран процессор {best_j} (несмотря на inf)", file=file)  #  Говорим, что выбрали несмотря на inf
        print("  загрузка задач:", sums, np.max(sums))
        print("  Распределение задач:", assignment, file=file)
        print("  Суммарная загрузка исполнителей:", sums, file=file)
        print("  Максимальная загрузка:", np.max(sums), file=file)
        results.append({"matrix_index": matrix_index, "assignment": assignment, "sums": sums, "max_load": np.max(sums), "criterion": "quadratic"})
    return results

def ALG_cubic(matrices, file):
    print("\nПрименение алгоритма Плотникова (степень: 3)...", file=file)
    print("\nПрименение алгоритма Плотникова (степень: 3)...")
    results = []
    for matrix_index, matrix in enumerate(matrices):
        print(f"\nМатрица {matrix_index}:")
        print(f"\nМатрица {matrix_index}:", file=file)
        M, N = matrix.shape
        sums = np.zeros(N)
        assignment = [-1] * M
        all_inf = False

        for i in range(M):
            best_j = -1
            min_cost = np.inf
            all_inf = True #  Флаг, что все inf

            print(f"  \nЗадача {i}:", file=file)
            for j in range(N):
                if matrix[i, j] != np.inf:
                    all_inf = False
                    #  Вычисляем общую "стоимость" системы, если назначить задачу процессору j
                    total_cost = 0
                    cost_components = []
                    for k in range(N):
                        if k == j:
                            processor_cost = (sums[k] + matrix[i, j])**3
                            cost_components.append(f"({sums[k]} + {matrix[i, j]})^3")
                        else:
                            processor_cost = sums[k]**3
                            cost_components.append(f"({sums[k]})^3")
                        total_cost += processor_cost
                    cost_string = " + ".join(cost_components) + f" = {total_cost}"
                    print(f"    изадачу {i} процессору {j}:", file=file)
                    print(f"      costs = {cost_string}", file=file)

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_j = j

            if not all_inf:
                sums[best_j] += matrix[i, best_j]
                assignment[i] = best_j
                print(f"  -> Выбираем процессор {best_j}: processors[{best_j}] += {matrix[i, best_j]}", file=file)
            else:  #  Если все процессоры inf, выбираем процессор с минимальной cost
                print("    Все процессоры недоступны. Выбираем лучший по стоимости...", file=file)
                best_j = -1
                min_cost = np.inf
                for j in range(N):  # Снова перебираем все процессоры
                    total_cost = 0
                    cost_components = []
                    for k in range(N):
                        if k == j:
                            processor_cost = (sums[k] + matrix[i, j])**3  # matrix[i, j] == inf
                            cost_components.append(f"({sums[k]} + {matrix[i, j]})^3")
                        else:
                            processor_cost = sums[k]**3
                            cost_components.append(f"({sums[k]})^3")
                        total_cost += processor_cost
                    cost_string = " + ".join(cost_components) + f" = {total_cost}"
                    print(f"    Процессор {j}: cost = {cost_string}", file=file)

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_j = j
                sums[best_j] += matrix[i, best_j]
                assignment[i] = best_j
                print(f"    Выбран процессор {best_j} (несмотря на inf)", file=file)  #  Говорим, что выбрали несмотря на inf
        print("  загрузка задач:", sums, np.max(sums))
        print("  Распределение задач:", assignment, file=file)
        print("  Суммарная загрузка исполнителей:", sums, file=file)
        print("  Максимальная загрузка:", np.max(sums), file=file)
        results.append({"matrix_index": matrix_index, "assignment": assignment, "sums": sums, "max_load": np.max(sums), "criterion": "cubic"})
    return results

N = int(input("N = (Количество исполнителей) "))
M = int(input("M = (Количество задач) "))
MICO = int(input("MICO = (кол-во матриц) "))
print("Введите границы диапазона для значений задач:")
t1 = int(input("t1 = (Минимум) "))
t2 = int(input("t2 = (Максимум) "))

# Создаем три матрицы (или запрашиваем ввод)
MATRICES = []
for i in range(MICO):
    print(f"\nВвод матрицы {i}:")
    T = create_random_matrix(N, M, t1, t2)
    # Замените следующую строку на
    print("Исходная матрица:\n", T)
    MATRICES.append(T)

while True:
    print("\nВыберите метод сортировки:")
    print("1) Сортировка по убыванию количества бесконечностей")
    print("2) Сортировка по убыванию количества бесконечностей и возрастанию суммы")
    print("3) Исходная матрица")
    print("4) Выход")
    method = int(input("Выберите метод сортировки: "))

    if method == 4:
        break

    # Открываем файл для записи
    with open("results.txt", "w") as file:
        print(f"Выбран метод сортировки: {method}", file=file)
        # Сортировка матриц
        SORTED_MATRICES = []
        for index, T in enumerate(MATRICES):
            S = sorting(method, T)
            print(f"\nОтсортированная матрица {index}:\n", S, file=file)
            SORTED_MATRICES.append(S)

        # Запускаем все алгоритмы и сохраняем результаты
        minimax_results = ALG_minimax(SORTED_MATRICES, file)
        quadratic_results = ALG_quadratic(SORTED_MATRICES, file)
        cubic_results = ALG_cubic(SORTED_MATRICES, file)

        # Собираем все результаты в один список
        all_results = minimax_results + quadratic_results + cubic_results

        # Подсчет побед
        wins = {"minimax": 0, "quadratic": 0, "cubic": 0}
        for matrix_index in range(MICO):
            min_max_load = np.inf
            best_criterion = None
            for result in all_results:
                if result["matrix_index"] == matrix_index:
                    if result["max_load"] < min_max_load:
                        min_max_load = result["max_load"]
                        best_criterion = result["criterion"]
            if best_criterion:
                wins[best_criterion] += 1

        # Сравнение и запись в файл
        for matrix_index in range(MICO):
            minimax_load = None
            quadratic_load = None
            cubic_load = None
            for result in all_results:
                if result["matrix_index"] == matrix_index and result["criterion"] == "minimax":
                    minimax_load = result["max_load"]
                elif result["matrix_index"] == matrix_index and result["criterion"] == "quadratic":
                    quadratic_load = result["max_load"]
                elif result["matrix_index"] == matrix_index and result["criterion"] == "cubic":
                    cubic_load = result["max_load"]

            winner = None #Определяем победителя
            if minimax_load is not None:
                winner = "minimax"
                winning_load = minimax_load
            if quadratic_load is not None and (winner is None or quadratic_load < winning_load):
                winner = "quadratic"
                winning_load = quadratic_load
            if cubic_load is not None and (winner is None or cubic_load < winning_load):
                winner = "cubic"
                winning_load = cubic_load

            if winner == "quadratic":
                print(f"\nДля матрицы {matrix_index} quadratic критерий лучше всех (max_load: {quadratic_load})", file=file)
            elif winner == "cubic":
                print(f"\nДля матрицы {matrix_index} cubic критерий лучше всех (max_load: {cubic_load})", file=file)
            elif winner == "minimax":
                print(f"\nДля матрицы {matrix_index} minimax критерий лучше всех (max_load: {minimax_load})", file=file)
            else:
                print(f"\nДля матрицы {matrix_index} все критерии равны или не удалось определить лучшего", file=file)

        # Вычисление и вывод средних значений максимальной загрузки для каждого критерия
        def calculate_average_max_load(results):
            total_max_load = 0
            count = 0
            for result in results:
                total_max_load += result["max_load"]
                count += 1
            return total_max_load / count

        average_minimax_load = calculate_average_max_load(minimax_results)
        average_quadratic_load = calculate_average_max_load(quadratic_results)
        average_cubic_load = calculate_average_max_load(cubic_results)

        print(f"\nСреднее значение максимальной загрузки (Minimax): {average_minimax_load}")
        print(f"Среднее значение максимальной загрузки (Quadratic): {average_quadratic_load}")
        print(f"Среднее значение максимальной загрузки (Cubic): {average_cubic_load}")
        print(f"\nКоличество побед:")
        print(f"  Minimax: {wins['minimax']}")
        print(f"  Quadratic: {wins['quadratic']}")
        print(f"  Cubic: {wins['cubic']}")