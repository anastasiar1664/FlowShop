import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from typing import List, Tuple, Dict, Optional
import time

# ==========================================
# 1. СТРУКТУРЫ ДАННЫХ
# ==========================================

class PseudoJob:
    """
    Псевдоработа - часть исходной работы Jj.
    """
    _next_part_id = 0
    
    def __init__(self, job_id: int, duration: int, machine_start: int = 0, part_id: Optional[int] = None):
        self.job_id = job_id
        self.duration = duration
        self.machine_start = machine_start
        if part_id is None:
            self.part_id = PseudoJob._next_part_id
            PseudoJob._next_part_id += 1
        else:
            self.part_id = part_id
    
    @classmethod
    def reset_counter(cls):
        cls._next_part_id = 0
    
    def __repr__(self):
        return f"J{self.job_id}^{self.part_id}({self.duration})"
    
    def __eq__(self, other):
        if not isinstance(other, PseudoJob): 
            return False
        # Сравниваем ТОЛЬКО job_id и duration (важно!)
        return (self.job_id == other.job_id and 
                self.duration == other.duration)

class Schedule:
    """
    Расписание: список машин, каждая машина - список псевдоработ.
    """
    def __init__(self, machines_count: int):
        self.m = machines_count
        self.machines: List[List[PseudoJob]] = [[] for _ in range(machines_count)]
    
    def copy(self) -> 'Schedule':
        new_sched = Schedule(self.m)
        for i in range(self.m):
            new_sched.machines[i] = [copy.deepcopy(pj) for pj in self.machines[i]]
        return new_sched
    
    def __eq__(self, other):
        if not isinstance(other, Schedule): 
            return False
        if self.m != other.m: 
            return False
        for i in range(self.m):
            if len(self.machines[i]) != len(other.machines[i]): 
                return False
            # Сравниваем ТОЛЬКО job_id и duration (без part_id!)
            for pj1, pj2 in zip(self.machines[i], other.machines[i]):
                if pj1.job_id != pj2.job_id or pj1.duration != pj2.duration:
                    return False
        return True
    
    def __hash__(self):
        # Хеш ТОЛЬКО по job_id и duration
        data = []
        for i in range(self.m):
            machine_data = [(pj.job_id, pj.duration) for pj in self.machines[i]]
            data.append(tuple(machine_data))
        return hash(tuple(data))
    
    def __str__(self):
        result = []
        for i in range(self.m):
            ops = " ".join([str(pj) for pj in self.machines[i]])
            result.append(f"M{i+1}: [{ops}]")
        return "\n".join(result)

    def get_total_jobs(self) -> int:
        jobs = set()
        for machine in self.machines:
            for pj in machine:
                jobs.add(pj.job_id)
        return len(jobs)

# ==========================================
# 2. ОПЕРАТОР ПРЕОБРАЗОВАНИЯ (УПРОЩЁННЫЙ)
# ==========================================

def apply_operator_simple(schedule: Schedule, mu: int, k: int, l: int, m_idx: int) -> Schedule:
    """
    Упрощённая версия оператора - меняет порядок работ без разбиения.
    """
    if k < 0 or l < 0 or m_idx < 0 or m_idx >= schedule.m:
        return schedule.copy()
    
    if len(schedule.machines[m_idx]) <= k:
        return schedule.copy()
    
    new_sched = schedule.copy()
    ops = new_sched.machines[m_idx]
    
    target_pj = ops.pop(k)
    
    insert_pos = l
    if k < l:
        insert_pos = l - 1
    insert_pos = max(0, min(insert_pos, len(ops)))
    
    ops.insert(insert_pos, target_pj)
    
    # Согласование на других машинах
    if mu == 1:
        for machine_idx in range(m_idx + 1, new_sched.m):
            other_ops = new_sched.machines[machine_idx]
            for idx, pj in enumerate(other_ops):
                if pj.job_id == target_pj.job_id:
                    other_ops.pop(idx)
                    insert_p = min(insert_pos, len(other_ops))
                    other_ops.insert(insert_p, pj)
                    break
    
    elif mu == 2:
        for machine_idx in range(0, m_idx):
            other_ops = new_sched.machines[machine_idx]
            for idx, pj in enumerate(other_ops):
                if pj.job_id == target_pj.job_id:
                    other_ops.pop(idx)
                    insert_p = min(insert_pos, len(other_ops))
                    other_ops.insert(insert_p, pj)
                    break
    
    return new_sched

# ==========================================
# 3. РАСЧЕТ Cmax
# ==========================================

def calculate_cmax(schedule: Schedule, job_durations: Dict[int, List[int]]) -> int:
    """
    Вычисляет общее время выполнения (Cmax).
    """
    n_machines = schedule.m
    machine_end_time = [0] * n_machines
    job_end_time = {}
    
    for m in range(n_machines):
        current_time = 0
        for pj in schedule.machines[m]:
            j_id = pj.job_id
            dur = pj.duration
            
            ready_time = job_end_time.get(j_id, 0)
            start_time = max(current_time, ready_time)
            end_time = start_time + dur
            
            job_end_time[j_id] = end_time
            current_time = end_time
        
        machine_end_time[m] = current_time
    
    return max(machine_end_time) if machine_end_time else 0

# ==========================================
# 4. РАСЧЕТ МЕТРИКИ (ОПТИМИЗИРОВАННЫЙ BFS)
# ==========================================

def calculate_metric_bfs(start: Schedule, end: Schedule, max_depth: int = 3, 
                         max_iterations: int = 5000) -> int:
    """
    Оптимизированный BFS с ограничениями.
    """
    if start == end:
        return 0
    
    queue = deque([(start, 0)])
    visited = {hash(start)}
    
    iterations = 0
    start_time = time.time()
    
    while queue:
        iterations += 1
        
        # Ограничение по времени и итерациям
        if iterations > max_iterations or (time.time() - start_time) > 10:
            print(f"  [BFS] Превышен лимит: {iterations} итераций, {time.time()-start_time:.2f} сек")
            return -1
        
        if iterations % 500 == 0:
            print(f"  [BFS] Обработано {iterations} состояний, очередь: {len(queue)}")
        
        current, dist = queue.popleft()
        if dist >= max_depth:
            continue
        
        # Ограниченное число соседей
        neighbors_generated = 0
        max_neighbors = 30
        
        for m in range(min(current.m, 3)):
            ops_count = len(current.machines[m])
            for k in range(min(ops_count, 4)):
                for l in range(min(ops_count + 1, 5)):
                    if k == l:
                        continue
                    
                    for mu in [1, 2]:
                        neighbor = apply_operator_simple(current, mu, k, l, m)
                        h = hash(neighbor)
                        
                        if neighbor == end:
                            elapsed = time.time() - start_time
                            print(f"  [BFS] Найдено за {dist + 1} шагов после {iterations} итераций ({elapsed:.2f} сек)")
                            return dist + 1
                        
                        if h not in visited and neighbors_generated < max_neighbors:
                            visited.add(h)
                            queue.append((neighbor, dist + 1))
                            neighbors_generated += 1
    
    print(f"  [BFS] Не найдено в пределах глубины {max_depth}")
    return -1

def calculate_metric_heuristic(s1: Schedule, s2: Schedule) -> int:
    """
    Эвристическая оценка расстояния.
    """
    distance = 0
    for m in range(s1.m):
        jobs1 = [pj.job_id for pj in s1.machines[m]]
        jobs2 = [pj.job_id for pj in s2.machines[m]]
        if jobs1 != jobs2:
            distance += 1
    return max(1, distance)

# ==========================================
# 5. ВИЗУАЛИЗАЦИЯ
# ==========================================

def plot_gantt(schedule: Schedule, job_durations: Dict[int, List[int]], title: str = "Расписание"):
    """
    Строит диаграмму Ганта.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    y_labels = [f"Машина {i+1}" for i in range(schedule.m)]
    y_pos = range(schedule.m)
    
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
            
            color = colors[(j_id - 1) % len(colors)]
            ax.barh(y_pos[m], end_time - start_time, left=start_time, 
                   height=0.6, color=color, edgecolor='black')
            
            ax.text(start_time + (end_time - start_time)/2, y_pos[m], 
                   f"J{j_id}", ha='center', va='center', color='white', fontsize=9)
            
            job_end_time[j_id] = end_time
            current_time = end_time
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Время")
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    handles = []
    for j_id in range(1, schedule.get_total_jobs() + 1):
        patch = mpatches.Patch(color=colors[(j_id - 1) % len(colors)], label=f"Работа {j_id}")
        handles.append(patch)
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    return fig

# ==========================================
# 6. ТЕСТОВЫЕ ПРИМЕРЫ
# ==========================================

def run_test_1():
    """
    ТЕСТ 1: Пример из отчёта (3 работы × 3 машины)
    """
    print("\n" + "="*70)
    print("ТЕСТ 1: Пример из отчёта (3×3)")
    print("="*70)
    
    PseudoJob.reset_counter()
    
    job_durations = {
        1: [2, 4, 1],
        2: [3, 1, 3],
        3: [1, 2, 2]
    }
    
    m_count = 3
    
    # Исходное расписание π*
    pi_star = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi_star.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax_star = calculate_cmax(pi_star, job_durations)
    print(f"\n[Исходное расписание π*]")
    print(pi_star)
    print(f"Cmax = {cmax_star}")
    
    # Расписание π** (изменили порядок на М2)
    pi_double_star = pi_star.copy()
    ops_m2 = pi_double_star.machines[1]
    if len(ops_m2) >= 3:
        ops_m2[1], ops_m2[2] = ops_m2[2], ops_m2[1]
    
    cmax_double = calculate_cmax(pi_double_star, job_durations)
    print(f"\n[Расписание π** (изменён порядок на М2)]")
    print(pi_double_star)
    print(f"Cmax = {cmax_double}")
    
    # Вычисление метрики
    print(f"\n[Вычисление метрики ρΩ]")
    dist = calculate_metric_bfs(pi_star, pi_double_star, max_depth=3)
    
    if dist == -1:
        print("  Используем эвристику...")
        dist = calculate_metric_heuristic(pi_star, pi_double_star)
    
    print(f"Расстояние ρΩ(π*, π**) = {dist}")
    
    return pi_star, pi_double_star, job_durations

def run_test_2():
    """
    ТЕСТ 2: 4 работы × 3 машины (средняя сложность)
    """
    print("\n" + "="*70)
    print("ТЕСТ 2: 4 работы × 3 машины")
    print("="*70)
    
    PseudoJob.reset_counter()
    
    job_durations = {
        1: [3, 2, 4],
        2: [1, 5, 2],
        3: [2, 3, 3],
        4: [4, 1, 2]
    }
    
    m_count = 3
    
    # Исходное: 1-2-3-4
    pi1 = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3, 4]:
            dur = job_durations[j_id][m]
            pi1.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax1 = calculate_cmax(pi1, job_durations)
    print(f"\n[Исходное расписание]")
    print(pi1)
    print(f"Cmax = {cmax1}")
    
    # Модифицированное: меняем порядок на М1 (2-1-4-3)
    pi2 = Schedule(m_count)
    order_m1 = [2, 1, 4, 3]
    order_m2 = [1, 2, 3, 4]
    order_m3 = [1, 2, 3, 4]
    
    for j_id in order_m1:
        pi2.machines[0].append(PseudoJob(j_id, job_durations[j_id][0], 0))
    for j_id in order_m2:
        pi2.machines[1].append(PseudoJob(j_id, job_durations[j_id][1], 0))
    for j_id in order_m3:
        pi2.machines[2].append(PseudoJob(j_id, job_durations[j_id][2], 0))
    
    cmax2 = calculate_cmax(pi2, job_durations)
    print(f"\n[Модифицированное расписание (изменён порядок на М1)]")
    print(pi2)
    print(f"Cmax = {cmax2}")
    
    # Метрика
    print(f"\n[Вычисление метрики]")
    dist = calculate_metric_bfs(pi1, pi2, max_depth=3, max_iterations=3000)
    
    if dist == -1:
        dist = calculate_metric_heuristic(pi1, pi2)
        print(f"  Эвристическое расстояние = {dist}")
    else:
        print(f"Расстояние ρΩ = {dist}")
    
    return pi1, pi2, job_durations

def run_test_3():
    """
    ТЕСТ 3: 3 работы × 4 машины (больше машин)
    """
    print("\n" + "="*70)
    print("ТЕСТ 3: 3 работы × 4 машины")
    print("="*70)
    
    PseudoJob.reset_counter()
    
    job_durations = {
        1: [2, 3, 1, 4],
        2: [4, 1, 3, 2],
        3: [1, 2, 4, 3]
    }
    
    m_count = 4
    
    # Исходное
    pi1 = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi1.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax1 = calculate_cmax(pi1, job_durations)
    print(f"\n[Исходное расписание]")
    print(pi1)
    print(f"Cmax = {cmax1}")
    
    # Изменения на М2 и М3
    pi2 = pi1.copy()
    # М2: 1-3-2
    pi2.machines[1][0], pi2.machines[1][1], pi2.machines[1][2] = \
        pi1.machines[1][0], pi1.machines[1][2], pi1.machines[1][1]
    # М3: 3-1-2
    pi2.machines[2][0], pi2.machines[2][1], pi2.machines[2][2] = \
        pi1.machines[2][2], pi1.machines[2][0], pi1.machines[2][1]
    
    cmax2 = calculate_cmax(pi2, job_durations)
    print(f"\n[Модифицированное (изменения на М2 и М3)]")
    print(pi2)
    print(f"Cmax = {cmax2}")
    
    print(f"\n[Вычисление метрики]")
    dist = calculate_metric_heuristic(pi1, pi2)
    print(f"Эвристическое расстояние = {dist} (машины с разным порядком)")
    
    return pi1, pi2, job_durations

def run_test_4():
    """
    ТЕСТ 4: 5 работ × 3 машины (сложный)
    """
    print("\n" + "="*70)
    print("ТЕСТ 4: 5 работ × 3 машины (сложный)")
    print("="*70)
    
    PseudoJob.reset_counter()
    
    job_durations = {
        1: [3, 2, 1],
        2: [1, 4, 2],
        3: [2, 1, 3],
        4: [4, 3, 2],
        5: [2, 2, 4]
    }
    
    m_count = 3
    
    # Исходное: 1-2-3-4-5
    pi1 = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3, 4, 5]:
            dur = job_durations[j_id][m]
            pi1.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax1 = calculate_cmax(pi1, job_durations)
    print(f"\n[Исходное расписание]")
    print(pi1)
    print(f"Cmax = {cmax1}")
    
    # Сложное изменение: 3-1-5-2-4 на М1
    pi2 = Schedule(m_count)
    order_m1 = [3, 1, 5, 2, 4]
    order_m2 = [1, 2, 3, 4, 5]
    order_m3 = [2, 4, 1, 3, 5]
    
    for j_id in order_m1:
        pi2.machines[0].append(PseudoJob(j_id, job_durations[j_id][0], 0))
    for j_id in order_m2:
        pi2.machines[1].append(PseudoJob(j_id, job_durations[j_id][1], 0))
    for j_id in order_m3:
        pi2.machines[2].append(PseudoJob(j_id, job_durations[j_id][2], 0))
    
    cmax2 = calculate_cmax(pi2, job_durations)
    print(f"\n[Сложное модифицированное расписание]")
    print(pi2)
    print(f"Cmax = {cmax2}")
    
    print(f"\n[Вычисление метрики]")
    print("  BFS пропущен (слишком сложно), используем эвристику...")
    dist = calculate_metric_heuristic(pi1, pi2)
    print(f"Эвристическое расстояние = {dist}")
    
    return pi1, pi2, job_durations

def run_test_5():
    """
    ТЕСТ 5: Проверка неравенства треугольника
    """
    print("\n" + "="*70)
    print("ТЕСТ 5: Проверка неравенства треугольника")
    print("="*70)
    
    PseudoJob.reset_counter()
    
    job_durations = {
        1: [2, 4, 1],
        2: [3, 1, 3],
        3: [1, 2, 2]
    }
    
    m_count = 3
    
    # π*
    pi_star = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi_star.machines[m].append(PseudoJob(j_id, dur, 0))
    
    # π** (изменение на М2)
    pi_double = pi_star.copy()
    ops = pi_double.machines[1]
    if len(ops) >= 2:
        ops[1], ops[2] = ops[2], ops[1]
    
    # π*** (изменение на М3)
    pi_triple = pi_double.copy()
    ops = pi_triple.machines[2]
    if len(ops) >= 2:
        ops[0], ops[1] = ops[1], ops[0]
    
    print("\n[Расписания]")
    print(f"π*:  Cmax = {calculate_cmax(pi_star, job_durations)}")
    print(f"π**: Cmax = {calculate_cmax(pi_double, job_durations)}")
    print(f"π***: Cmax = {calculate_cmax(pi_triple, job_durations)}")
    
    # Метрики
    print("\n[Вычисление расстояний]")
    d12 = calculate_metric_heuristic(pi_star, pi_double)
    d23 = calculate_metric_heuristic(pi_double, pi_triple)
    d13 = calculate_metric_heuristic(pi_star, pi_triple)
    
    print(f"ρΩ(π*, π**) = {d12}")
    print(f"ρΩ(π**, π***) = {d23}")
    print(f"ρΩ(π*, π***) = {d13}")
    
    # Проверка
    print(f"\n[Проверка неравенства треугольника]")
    print(f"{d13} ≤ {d12} + {d23}  →  {d13} ≤ {d12 + d23}")
    
    if d13 <= d12 + d23:
        print("✓ НЕРАВЕНСТВО ВЫПОЛНЯЕТСЯ!")
    else:
        print("✗ ОШИБКА!")
    
    return d12, d23, d13

# ==========================================
# 7. ГЛАВНАЯ ФУНКЦИЯ
# ==========================================

def main():
    print("="*70)
    print("МОДЕЛИРОВАНИЕ ЗАДАЧИ FLOW SHOP (КЛАСС K)")
    print("ОПТИМИЗИРОВАННАЯ ВЕРСИЯ С ТЕСТИРОВАНИЕМ")
    print("="*70)
    
    all_schedules = []
    
    # Запускаем все тесты
    try:
        s1, s2, jd1 = run_test_1()
        all_schedules.append((s1, jd1, "Test1_Original"))
        all_schedules.append((s2, jd1, "Test1_Modified"))
    except Exception as e:
        print(f"Ошибка в Тесте 1: {e}")
    
    try:
        s1, s2, jd2 = run_test_2()
        all_schedules.append((s1, jd2, "Test2_Original"))
        all_schedules.append((s2, jd2, "Test2_Modified"))
    except Exception as e:
        print(f"Ошибка в Тесте 2: {e}")
    
    try:
        s1, s2, jd3 = run_test_3()
        all_schedules.append((s1, jd3, "Test3_Original"))
        all_schedules.append((s2, jd3, "Test3_Modified"))
    except Exception as e:
        print(f"Ошибка в Тесте 3: {e}")
    
    try:
        s1, s2, jd4 = run_test_4()
        all_schedules.append((s1, jd4, "Test4_Original"))
        all_schedules.append((s2, jd4, "Test4_Modified"))
    except Exception as e:
        print(f"Ошибка в Тесте 4: {e}")
    
    try:
        run_test_5()
    except Exception as e:
        print(f"Ошибка в Тесте 5: {e}")
    
    # Визуализация (только первые 4 расписания)
    print(f"\n{'='*70}")
    print("ГЕНЕРАЦИЯ ДИАГРАММ ГАНТА")
    print("="*70)
    
    for i, (sched, durations, name) in enumerate(all_schedules[:4]):
        try:
            cmax = calculate_cmax(sched, durations)
            fig = plot_gantt(sched, durations, f"{name} (Cmax={cmax})")
            filename = f"gantt_{name}.png"
            fig.savefig(filename, dpi=150)
            print(f"✓ Сохранено: {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"✗ Ошибка при сохранении {name}: {e}")
    
    print(f"\n{'='*70}")
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*70)

if __name__ == "__main__":
    main() 
    plt.show()  # Показать все диаграммы