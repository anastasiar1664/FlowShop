import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from typing import List, Tuple, Dict, Optional

# ==========================================
# 1. СТРУКТУРЫ ДАННЫХ
# ==========================================

class PseudoJob:
    """
    Псевдоработа - часть исходной работы Jj.
    """
    def __init__(self, job_id: int, part_id: int, duration: int, machine_start: int):
        self.job_id = job_id        # ID исходной работы (1, 2, 3...)
        self.part_id = part_id      # ID части (0, 1, 2...)
        self.duration = duration    # Длительность этой части
        self.machine_start = machine_start # С какой машины начинается эта псевдоработа
    
    def __repr__(self):
        return f"J{self.job_id}^{self.part_id}({self.duration})"
    
    def __eq__(self, other):
        if not isinstance(other, PseudoJob): return False
        return (self.job_id == other.job_id and 
                self.part_id == other.part_id and 
                self.duration == other.duration)

class Schedule:
    """
    Расписание: список машин, каждая машина - список псевдоработ.
    """
    def __init__(self, machines_count: int):
        self.m = machines_count
        # machines[i] = список псевдоработ на машине i
        self.machines: List[List[PseudoJob]] = [[] for _ in range(machines_count)]
    
    def copy(self) -> 'Schedule':
        new_sched = Schedule(self.m)
        for i in range(self.m):
            new_sched.machines[i] = [copy.deepcopy(pj) for pj in self.machines[i]]
        return new_sched
    
    def __eq__(self, other):
        if not isinstance(other, Schedule): return False
        if self.m != other.m: return False
        for i in range(self.m):
            if len(self.machines[i]) != len(other.machines[i]): return False
            for pj1, pj2 in zip(self.machines[i], other.machines[i]):
                if pj1 != pj2: return False
        return True
    
    def __hash__(self):
        # Хеш для хранения в множествах (BFS)
        data = []
        for i in range(self.m):
            machine_data = [(pj.job_id, pj.part_id, pj.duration) for pj in self.machines[i]]
            data.append(tuple(machine_data))
        return hash(tuple(data))
    
    def __str__(self):
        result = []
        for i in range(self.m):
            ops = " ".join([str(pj) for pj in self.machines[i]])
            result.append(f"M{i+1}: [{ops}]")
        return "\n".join(result)

    def get_total_jobs(self) -> int:
        """Возвращает количество уникальных исходных работ."""
        jobs = set()
        for machine in self.machines:
            for pj in machine:
                jobs.add(pj.job_id)
        return len(jobs)

# ==========================================
# 2. ОПЕРАТОР ПРЕОБРАЗОВАНИЯ Ω(μ, k, l, m)
# ==========================================

def apply_operator(schedule: Schedule, mu: int, k: int, l: int, m_idx: int) -> Schedule:
    """
    Применяет оператор Ω(μ, k, l, m) к расписанию.
    
    Параметры:
    - mu: 1 (начиная с машины m) или 2 (заканчивая машиной m)
    - k: позиция исходной псевдоработы (на машине m_idx)
    - l: целевая позиция вставки
    - m_idx: номер машины (0-based), задающая границу
    """
    new_sched = schedule.copy()
    
    if m_idx < 0 or m_idx >= new_sched.m:
        return new_sched
    
    ops = new_sched.machines[m_idx]
    if len(ops) == 0 or k < 0 or k >= len(ops):
        return new_sched
    
    # Извлекаем псевдоработу
    target_pj = ops.pop(k)
    
    # Корректируем l, если удалили элемент перед позицией вставки
    insert_l = l
    if k < l:
        insert_l = l - 1
    insert_l = max(0, min(insert_l, len(ops)))
    
    if mu == 1:
        # Первый род: изменение порядка начиная с машины m
        # Вставляем на машине m, и на всех последующих (m, m+1, ...) порядок согласуется
        ops.insert(insert_l, target_pj)
        
        # Для согласованности (упрощенно) применяем то же перемещение на машинах > m_idx
        for i in range(m_idx + 1, new_sched.m):
            other_ops = new_sched.machines[i]
            # Ищем соответствующую часть той же работы
            for idx, pj in enumerate(other_ops):
                if pj.job_id == target_pj.job_id:
                    # Перемещаем её на аналогичную позицию относительно других частей
                    other_ops.pop(idx)
                    # Пытаемся сохранить относительный порядок
                    target_idx = min(insert_l, len(other_ops))
                    other_ops.insert(target_idx, pj)
                    break
                    
    elif mu == 2:
        # Второй род: изменение порядка заканчивая машиной m
        # Вставляем на машине m, и на всех предыдущих (0, ..., m) порядок согласуется
        ops.insert(insert_l, target_pj)
        
        # Применяем на машинах < m_idx
        for i in range(0, m_idx):
            other_ops = new_sched.machines[i]
            for idx, pj in enumerate(other_ops):
                if pj.job_id == target_pj.job_id:
                    other_ops.pop(idx)
                    target_idx = min(insert_l, len(other_ops))
                    other_ops.insert(target_idx, pj)
                    break
    
    return new_sched

def split_job_at_machine(schedule: Schedule, job_id: int, machine_idx: int, split_duration: int) -> Schedule:
    """
    Вспомогательная функция: разбивает псевдоработу на две части на указанной машине.
    Нужно для создания расписаний с псевдоработами (как в примере отчета).
    """
    new_sched = schedule.copy()
    for m in range(new_sched.m):
        ops = new_sched.machines[m]
        for idx, pj in enumerate(ops):
            if pj.job_id == job_id and pj.duration > split_duration and m >= machine_idx:
                # Разбиваем
                part1 = PseudoJob(pj.job_id, pj.part_id * 2, split_duration, pj.machine_start)
                part2 = PseudoJob(pj.job_id, pj.part_id * 2 + 1, pj.duration - split_duration, machine_idx)
                ops.pop(idx)
                ops.insert(idx, part1)
                ops.insert(idx + 1, part2)
                break
    return new_sched

# ==========================================
# 3. РАСЧЕТ Cmax (ДЛИНА РАСПИСАНИЯ)
# ==========================================

def calculate_cmax(schedule: Schedule, job_durations: Dict[int, List[int]]) -> int:
    """
    Вычисляет общее время выполнения (Cmax) с учетом технологического предшествования.
    """
    n_machines = schedule.m
    # Время окончания последней операции на каждой машине
    machine_end_time = [0] * n_machines
    # Время окончания последней операции для каждой работы (по номеру работы)
    job_end_time = {}
    
    # Проходим по машинам последовательно
    for m in range(n_machines):
        current_time = 0
        for pj in schedule.machines[m]:
            j_id = pj.job_id
            dur = pj.duration
            
            # Работа не может начаться на машине m, пока не закончится на машине m-1
            ready_time = job_end_time.get(j_id, 0)
            start_time = max(current_time, ready_time)
            end_time = start_time + dur
            
            job_end_time[j_id] = end_time
            current_time = end_time
        
        machine_end_time[m] = current_time
    
    return max(machine_end_time) if machine_end_time else 0

# ==========================================
# 4. РАСЧЕТ МЕТРИКИ ρΩ (BFS)
# ==========================================

def calculate_metric_bfs(start: Schedule, end: Schedule, max_depth: int = 5) -> int:
    """
    Вычисляет расстояние между расписаниями через поиск в ширину.
    """
    if start == end:
        return 0
    
    queue = deque([(start, 0)])
    visited = {hash(start)}
    
    while queue:
        current, dist = queue.popleft()
        if dist >= max_depth:
            continue
        
        # Генерируем соседей
        for m in range(current.m):
            ops_count = len(current.machines[m])
            for k in range(ops_count):
                for l in range(ops_count + 1):
                    for mu in [1, 2]:
                        if k == l: continue # Пустой оператор
                        
                        neighbor = apply_operator(current, mu, k, l, m)
                        h = hash(neighbor)
                        
                        if neighbor == end:
                            return dist + 1
                        
                        if h not in visited:
                            visited.add(h)
                            queue.append((neighbor, dist + 1))
    
    return -1  # Не найдено в пределах глубины

# ==========================================
# 5. ВИЗУАЛИЗАЦИЯ (ДИАГРАММА ГАНТА)
# ==========================================

def plot_gantt(schedule: Schedule, job_durations: Dict[int, List[int]], title: str = "Расписание"):
    """
    Строит диаграмму Ганта для расписания.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    y_labels = [f"Машина {i+1}" for i in range(schedule.m)]
    y_pos = range(schedule.m)
    
    # Рассчитываем времена начала и конца для каждой операции
    machine_end_time = [0] * schedule.m
    job_end_time = {}
    
    for m in range(schedule.m):
        current_time = 0
        for pj in schedule.machines[m]:
            j_id = pj.job_id
            dur = pj.duration
            
            ready_time = job_end_time.get(j_id, 0)
            start_time = max(current_time, ready_time)
            end_time = start_time + dur
            
            # Рисуем прямоугольник
            color = colors[(j_id - 1) % len(colors)]
            ax.barh(y_pos[m], end_time - start_time, left=start_time, 
                   height=0.6, color=color, edgecolor='black', label=f"J{j_id}")
            
            # Текст внутри
            ax.text(start_time + (end_time - start_time)/2, y_pos[m], 
                   f"J{j_id}", ha='center', va='center', color='white', fontsize=9)
            
            job_end_time[j_id] = end_time
            current_time = end_time
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Время")
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Легенда (уникальные работы)
    handles = []
    for j_id in range(1, schedule.get_total_jobs() + 1):
        patch = mpatches.Patch(color=colors[(j_id - 1) % len(colors)], label=f"Работа {j_id}")
        handles.append(patch)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    return fig

# ==========================================
# 6. ГЛАВНАЯ ФУНКЦИЯ (ПРИМЕР ИЗ ОТЧЕТА)
# ==========================================

def main():
    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ ЗАДАЧИ FLOW SHOP (КЛАСС K)")
    print("=" * 60)
    
    # Данные из Таблицы 1 отчета (3 работы, 3 машины)
    # job_durations[job_id] = [t1, t2, t3]
    job_durations = {
        1: [2, 4, 1],
        2: [3, 1, 3],
        3: [1, 2, 2]
    }
    
    m_count = 3
    
    # --- Исходное расписание π* (перестановочное, порядок 1-2-3) ---
    pi_star = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi_star.machines[m].append(PseudoJob(j_id, 0, dur, 0))
    
    cmax_star = calculate_cmax(pi_star, job_durations)
    print(f"\n[Исходное расписание π*]")
    print(pi_star)
    print(f"Cmax = {cmax_star}")
    
    # --- Расписание π** (с изменением на машине 2) ---
    pi_double_star = pi_star.copy()
    # Меняем порядок на машине 2 (индекс 1): 1, 3, 2 вместо 1, 2, 3
    ops_m2 = pi_double_star.machines[1]
    # Удаляем работу 2 и вставляем после работы 3
    job2 = ops_m2.pop(1)  # J2
    ops_m2.insert(2, job2)  # Вставляем после J3
    
    cmax_double = calculate_cmax(pi_double_star, job_durations)
    print(f"\n[Расписание π** (изменен порядок на М2)]")
    print(pi_double_star)
    print(f"Cmax = {cmax_double}")
    
    # --- Расчет метрики ---
    print(f"\n[Вычисление метрики ρΩ]")
    dist = calculate_metric_bfs(pi_star, pi_double_star, max_depth=5)
    print(f"Расстояние ρΩ(π*, π**) = {dist}")
    print(f"(Ожидается 1, так как это один шаг оператора)")
    
    # --- Проверка неравенства треугольника ---
    # Создадим третье расписание π*** (еще одно изменение)
    pi_triple_star = pi_double_star.copy()
    ops_m3 = pi_triple_star.machines[2]
    if len(ops_m3) >= 2:
        ops_m3[0], ops_m3[1] = ops_m3[1], ops_m3[0]
    
    cmax_triple = calculate_cmax(pi_triple_star, job_durations)
    dist_12 = dist  # π* -> π**
    dist_23 = calculate_metric_bfs(pi_double_star, pi_triple_star, max_depth=5)
    dist_13 = calculate_metric_bfs(pi_star, pi_triple_star, max_depth=5)
    
    print(f"\n[Проверка неравенства треугольника]")
    print(f"ρΩ(π*, π**) = {dist_12}")
    print(f"ρΩ(π**, π***) = {dist_23}")
    print(f"ρΩ(π*, π***) = {dist_13}")
    print(f"Неравенство: {dist_13} ≤ {dist_12} + {dist_23}  ->  {dist_13} ≤ {dist_12 + dist_23}")
    if dist_13 <= dist_12 + dist_23:
        print("✓ Неравенство треугольника выполняется!")
    else:
        print("✗ Ошибка вычислений!")
    
    # --- Визуализация ---
    print(f"\n[Генерация диаграмм Ганта...]")
    fig1 = plot_gantt(pi_star, job_durations, f"Исходное π* (Cmax={cmax_star})")
    fig2 = plot_gantt(pi_double_star, job_durations, f"Модифицированное π** (Cmax={cmax_double})")
    
    fig1.savefig("gantt_original.png", dpi=150)
    fig2.savefig("gantt_modified.png", dpi=150)
    print("Диаграммы сохранены как 'gantt_original.png' и 'gantt_modified.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)

if __name__ == "__main__":
    main()