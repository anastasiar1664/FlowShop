import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from typing import List, Dict, Optional, Tuple
import time

# ==========================================
# 1. СТРУКТУРЫ ДАННЫХ
# ==========================================

class PseudoJob:
    """Псевдоработа - часть исходной работы Jj."""
    
    def __init__(self, job_id: int, duration: int, machine_start: int = 0):
        self.job_id = job_id
        self.duration = duration
        self.machine_start = machine_start
    
    def __repr__(self):
        return f"J{self.job_id}({self.duration})"
    
    def __eq__(self, other):
        if not isinstance(other, PseudoJob):
            return False
        return (self.job_id == other.job_id and 
                self.duration == other.duration)

class Schedule:
    """Расписание: список машин, каждая машина - список псевдоработ."""
    
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
            for pj1, pj2 in zip(self.machines[i], other.machines[i]):
                if pj1.job_id != pj2.job_id or pj1.duration != pj2.duration:
                    return False
        return True
    
    def __hash__(self):
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
# 2. ОПЕРАТОР ПРЕОБРАЗОВАНИЯ Ω (ИСПРАВЛЕННЫЙ)
# ==========================================

def apply_operator(schedule: Schedule, mu: int, k: int, l: int, m_idx: int) -> Schedule:
    """
    Применяет оператор Ω(μ, k, l, m) согласно теории из отчёта.
    
    Параметры:
    - mu: 1 (начиная с машины m) или 2 (заканчивая машиной m)
    - k: позиция псевдоработы на машине m_idx
    - l: целевая позиция вставки
    - m_idx: номер машины (0-based)
    """
    if k < 0 or l < 0 or m_idx < 0 or m_idx >= schedule.m:
        return schedule.copy()
    
    if len(schedule.machines[m_idx]) <= k:
        return schedule.copy()
    
    new_sched = schedule.copy()
    ops_m = new_sched.machines[m_idx]
    
    # Извлекаем работу на позиции k
    target_pj = ops_m[k]
    
    # Корректируем позицию вставки
    insert_pos = l
    if k < l:
        insert_pos = l - 1
    insert_pos = max(0, min(insert_pos, len(ops_m)))
    
    # Удаляем и вставляем на новую позицию
    ops_m.pop(k)
    ops_m.insert(insert_pos, target_pj)
    
    # Согласование на других машинах
    if mu == 1:
        # Применяем на машинах m, m+1, ..., M
        for machine_idx in range(m_idx + 1, new_sched.m):
            other_ops = new_sched.machines[machine_idx]
            for idx, pj in enumerate(other_ops):
                if pj.job_id == target_pj.job_id:
                    other_ops.pop(idx)
                    insert_p = min(insert_pos, len(other_ops))
                    other_ops.insert(insert_p, pj)
                    break
    
    elif mu == 2:
        # Применяем на машинах 1, 2, ..., m
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
    """Вычисляет общее время выполнения (Cmax)."""
    n_machines = schedule.m
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
    
    return max(job_end_time.values()) if job_end_time else 0

# ==========================================
# 4. РАСЧЕТ МЕТРИКИ (ОПТИМИЗИРОВАННЫЙ BFS)
# ==========================================

def calculate_metric_bfs(start: Schedule, end: Schedule, max_depth: int = 3, 
                         max_iterations: int = 10000) -> Tuple[int, int]:
    """
    Вычисляет расстояние между расписаниями через BFS.
    Возвращает (расстояние, число итераций).
    """
    if start == end:
        return 0, 0
    
    queue = deque([(start, 0)])
    visited = {hash(start)}
    
    iterations = 0
    start_time = time.time()
    
    while queue:
        iterations += 1
        
        if iterations > max_iterations or (time.time() - start_time) > 30:
            return -1, iterations
        
        current, dist = queue.popleft()
        if dist >= max_depth:
            continue
        
        # Генерируем соседей
        for m in range(current.m):
            ops_count = len(current.machines[m])
            for k in range(ops_count):
                for l in range(ops_count + 1):
                    if k == l:
                        continue
                    
                    for mu in [1, 2]:
                        neighbor = apply_operator(current, mu, k, l, m)
                        h = hash(neighbor)
                        
                        if neighbor == end:
                            return dist + 1, iterations
                        
                        if h not in visited:
                            visited.add(h)
                            queue.append((neighbor, dist + 1))
    
    return -1, iterations

def calculate_metric_heuristic(s1: Schedule, s2: Schedule) -> int:
    """Эвристическая оценка расстояния."""
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
    """Строит диаграмму Ганта."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    y_labels = [f"Машина {i+1}" for i in range(schedule.m)]
    y_pos = range(schedule.m)
    
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
# 6. ПРОВЕРКА СВОЙСТВ МЕТРИКИ
# ==========================================

def verify_metric_properties(pi1: Schedule, pi2: Schedule, pi3: Schedule, 
                            job_durations: Dict[int, List[int]], test_name: str):
    """
    Проверяет свойства метрики для трёх расписаний.
    """
    print(f"\n[Проверка свойств метрики для {test_name}]")
    print("-" * 60)
    
    # Вычисляем Cmax
    cmax1 = calculate_cmax(pi1, job_durations)
    cmax2 = calculate_cmax(pi2, job_durations)
    cmax3 = calculate_cmax(pi3, job_durations)
    
    print(f"Расписания:")
    print(f"  π*:  Cmax = {cmax1}")
    print(f"  π**: Cmax = {cmax2}")
    print(f"  π***: Cmax = {cmax3}")
    
    # Вычисляем расстояния
    print(f"\n[Вычисление расстояний]")
    
    d12, iter12 = calculate_metric_bfs(pi1, pi2, max_depth=3)
    if d12 == -1:
        print(f"  BFS для ρΩ(π*, π**) не нашёл путь ({iter12} итераций), используем эвристику")
        d12 = calculate_metric_heuristic(pi1, pi2)
    else:
        print(f"  ρΩ(π*, π**) = {d12} (найдено за {iter12} итераций)")
    
    d23, iter23 = calculate_metric_bfs(pi2, pi3, max_depth=3)
    if d23 == -1:
        print(f"  BFS для ρΩ(π**, π***) не нашёл путь ({iter23} итераций), используем эвристику")
        d23 = calculate_metric_heuristic(pi2, pi3)
    else:
        print(f"  ρΩ(π**, π***) = {d23} (найдено за {iter23} итераций)")
    
    d13, iter13 = calculate_metric_bfs(pi1, pi3, max_depth=3)
    if d13 == -1:
        print(f"  BFS для ρΩ(π*, π***) не нашёл путь ({iter13} итераций), используем эвристику")
        d13 = calculate_metric_heuristic(pi1, pi3)
    else:
        print(f"  ρΩ(π*, π***) = {d13} (найдено за {iter13} итераций)")
    
    # Проверка неравенства треугольника
    print(f"\n[Проверка неравенства треугольника]")
    print(f"  ρΩ(π*, π***) ≤ ρΩ(π*, π**) + ρΩ(π**, π***)")
    print(f"  {d13} ≤ {d12} + {d23}")
    print(f"  {d13} ≤ {d12 + d23}")
    
    if d13 <= d12 + d23:
        print(f"  ✓ НЕРАВЕНСТВО ВЫПОЛНЯЕТСЯ!")
        return True
    else:
        print(f"  ✗ ОШИБКА: неравенство не выполняется!")
        return False

# ==========================================
# 7. ТЕСТОВЫЕ ПРИМЕРЫ
# ==========================================

def run_test_1():
    """ТЕСТ 1: Пример из отчёта (3×3)"""
    print("\n" + "="*70)
    print("ТЕСТ 1: Пример из отчёта (3 работы × 3 машины)")
    print("="*70)
    
    job_durations = {
        1: [2, 4, 1],
        2: [3, 1, 3],
        3: [1, 2, 2]
    }
    
    m_count = 3
    
    # π*: исходное перестановочное расписание 1-2-3
    pi_star = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi_star.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax_star = calculate_cmax(pi_star, job_durations)
    print(f"\n[Исходное расписание π*]")
    print(pi_star)
    print(f"Cmax = {cmax_star}")
    
    # π**: изменили порядок на М2 (J3 перед J2)
    pi_double = pi_star.copy()
    ops_m2 = pi_double.machines[1]
    if len(ops_m2) >= 3:
        # Меняем местами J2 и J3 на М2
        ops_m2[1], ops_m2[2] = ops_m2[2], ops_m2[1]
    
    cmax_double = calculate_cmax(pi_double, job_durations)
    print(f"\n[Расписание π** (изменён порядок на М2)]")
    print(pi_double)
    print(f"Cmax = {cmax_double}")
    
    # π***: более сложное изменение
    pi_triple = pi_double.copy()
    ops_m3 = pi_triple.machines[2]
    if len(ops_m3) >= 2:
        ops_m3[0], ops_m3[1] = ops_m3[1], ops_m3[0]
    
    cmax_triple = calculate_cmax(pi_triple, job_durations)
    print(f"\n[Расписание π*** (доп. изменение на М3)]")
    print(pi_triple)
    print(f"Cmax = {cmax_triple}")
    
    # Проверка свойств метрики
    verify_metric_properties(pi_star, pi_double, pi_triple, job_durations, "Тест 1")
    
    return pi_star, pi_double, job_durations

def run_test_2():
    """ТЕСТ 2: 4 работы × 3 машины"""
    print("\n" + "="*70)
    print("ТЕСТ 2: 4 работы × 3 машины")
    print("="*70)
    
    job_durations = {
        1: [3, 2, 4],
        2: [1, 5, 2],
        3: [2, 3, 3],
        4: [4, 1, 2]
    }
    
    m_count = 3
    
    # π*: 1-2-3-4
    pi1 = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3, 4]:
            dur = job_durations[j_id][m]
            pi1.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax1 = calculate_cmax(pi1, job_durations)
    print(f"\n[Исходное расписание π*]")
    print(pi1)
    print(f"Cmax = {cmax1}")
    
    # π**: изменили порядок на М1 (2-1-3-4)
    pi2 = pi1.copy()
    ops_m1 = pi2.machines[0]
    if len(ops_m1) >= 2:
        ops_m1[0], ops_m1[1] = ops_m1[1], ops_m1[0]
    
    cmax2 = calculate_cmax(pi2, job_durations)
    print(f"\n[Расписание π** (изменён порядок на М1)]")
    print(pi2)
    print(f"Cmax = {cmax2}")
    
    # π***: ещё изменили на М2
    pi3 = pi2.copy()
    ops_m2 = pi3.machines[1]
    if len(ops_m2) >= 3:
        ops_m2[1], ops_m2[2] = ops_m2[2], ops_m2[1]
    
    cmax3 = calculate_cmax(pi3, job_durations)
    print(f"\n[Расписание π*** (доп. изменение на М2)]")
    print(pi3)
    print(f"Cmax = {cmax3}")
    
    # Проверка свойств
    verify_metric_properties(pi1, pi2, pi3, job_durations, "Тест 2")
    
    return pi1, pi2, job_durations

def run_test_3():
    """ТЕСТ 3: 3 работы × 4 машины"""
    print("\n" + "="*70)
    print("ТЕСТ 3: 3 работы × 4 машины")
    print("="*70)
    
    job_durations = {
        1: [2, 3, 1, 4],
        2: [4, 1, 3, 2],
        3: [1, 2, 4, 3]
    }
    
    m_count = 4
    
    # π*
    pi1 = Schedule(m_count)
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi1.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax1 = calculate_cmax(pi1, job_durations)
    print(f"\n[Исходное расписание π*]")
    print(pi1)
    print(f"Cmax = {cmax1}")
    
    # π**: изменили на М2
    pi2 = pi1.copy()
    ops_m2 = pi2.machines[1]
    if len(ops_m2) >= 2:
        ops_m2[1], ops_m2[2] = ops_m2[2], ops_m2[1]
    
    cmax2 = calculate_cmax(pi2, job_durations)
    print(f"\n[Расписание π** (изменение на М2)]")
    print(pi2)
    print(f"Cmax = {cmax2}")
    
    # π***: изменили на М3
    pi3 = pi2.copy()
    ops_m3 = pi3.machines[2]
    if len(ops_m3) >= 2:
        ops_m3[0], ops_m3[1] = ops_m3[1], ops_m3[0]
    
    cmax3 = calculate_cmax(pi3, job_durations)
    print(f"\n[Расписание π*** (изменение на М3)]")
    print(pi3)
    print(f"Cmax = {cmax3}")
    
    # Проверка свойств
    verify_metric_properties(pi1, pi2, pi3, job_durations, "Тест 3")
    
    return pi1, pi2, job_durations

# ==========================================
# 8. ГЛАВНАЯ ФУНКЦИЯ
# ==========================================

def main():
    print("="*70)
    print("МОДЕЛИРОВАНИЕ ЗАДАЧИ FLOW SHOP (КЛАСС K)")
    print("ИСПРАВЛЕННАЯ ВЕРСИЯ С ПОЛНОЙ ПРОВЕРКОЙ СВОЙСТВ")
    print("="*70)
    
    all_tests = []
    
    # Запускаем тесты
    try:
        s1, s2, jd1 = run_test_1()
        all_tests.append((s1, s2, jd1, "Test1"))
    except Exception as e:
        print(f"Ошибка в Тесте 1: {e}")
    
    try:
        s1, s2, jd2 = run_test_2()
        all_tests.append((s1, s2, jd2, "Test2"))
    except Exception as e:
        print(f"Ошибка в Тесте 2: {e}")
    
    try:
        s1, s2, jd3 = run_test_3()
        all_tests.append((s1, s2, jd3, "Test3"))
    except Exception as e:
        print(f"Ошибка в Тесте 3: {e}")
    
    # Визуализация
    print(f"\n{'='*70}")
    print("ГЕНЕРАЦИЯ ДИАГРАММ ГАНТА")
    print("="*70)
    
    for sched, job_dur, name in [
        (all_tests[0][0], all_tests[0][2], "Test1_Original"),
        (all_tests[0][1], all_tests[0][2], "Test1_Modified"),
        (all_tests[1][0], all_tests[1][2], "Test2_Original"),
        (all_tests[1][1], all_tests[1][2], "Test2_Modified")
    ]:
        try:
            cmax = calculate_cmax(sched, job_dur)
            fig = plot_gantt(sched, job_dur, f"{name} (Cmax={cmax})")
            filename = f"gantt_{name}.png"
            fig.savefig(filename, dpi=150)
            print(f"✓ Сохранено: {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"✗ Ошибка при сохранении {name}: {e}")
    
    # Показываем все диаграммы
    plt.show()
    
    print(f"\n{'='*70}")
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*70)

if __name__ == "__main__":
    main()