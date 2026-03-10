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
    _next_part_id = 0  # Счетчик для уникальных ID частей
    
    def __init__(self, job_id: int, duration: int, machine_start: int = 0, part_id: Optional[int] = None):
        self.job_id = job_id        # ID исходной работы (1, 2, 3...)
        self.duration = duration    # Длительность этой части
        self.machine_start = machine_start # С какой машины начинается эта псевдоработа
        # part_id генерируется автоматически для уникальности
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
        # Хеш для хранения в множествах (BFS)
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
    if k < 0 or l < 0 or m_idx < 0 or m_idx >= schedule.m:
        return schedule.copy()
    
    if len(schedule.machines[m_idx]) <= k:
        return schedule.copy()
    
    new_sched = schedule.copy()
    
    # Извлекаем псевдоработу с машины m_idx на позиции k
    target_pj = new_sched.machines[m_idx][k]
    
    if mu == 1:
        # Первый род: изменение порядка НАЧИНАЯ с машины m_idx
        # Разрезаем работу на машине m_idx и перемещаем часть
        # Применяем изменения на машинах m_idx, m_idx+1, ..., M-1
        
        # На машине m_idx: разрезаем работу и вставляем вторую часть на позицию l
        ops_m = new_sched.machines[m_idx]
        
        # Создаем две псевдоработы из target_pj
        # Первая часть остается на позиции k
        part1_duration = target_pj.duration // 2
        part2_duration = target_pj.duration - part1_duration
        
        if part1_duration > 0 and part2_duration > 0:
            # Заменяем исходную работу на две части
            part1 = PseudoJob(target_pj.job_id, part1_duration, target_pj.machine_start)
            part2 = PseudoJob(target_pj.job_id, part2_duration, m_idx)
            
            # Удаляем исходную и вставляем две части
            ops_m.pop(k)
            ops_m.insert(k, part1)
            
            # Определяем позицию для вставки второй части
            insert_pos = l if l <= len(ops_m) else len(ops_m)
            if k < insert_pos:
                insert_pos -= 1  # Корректируем, так как удалили элемент
            ops_m.insert(insert_pos, part2)
        
        # На последующих машинах (m_idx+1, ..., M-1) также применяем изменение
        for machine_idx in range(m_idx + 1, new_sched.m):
            ops = new_sched.machines[machine_idx]
            # Находим работу с тем же job_id
            for idx, pj in enumerate(ops):
                if pj.job_id == target_pj.job_id:
                    # Разрезаем и перемещаем аналогично
                    part1_dur = pj.duration // 2
                    part2_dur = pj.duration - part1_dur
                    
                    if part1_dur > 0 and part2_dur > 0:
                        part1 = PseudoJob(pj.job_id, part1_dur, pj.machine_start)
                        part2 = PseudoJob(pj.job_id, part2_dur, m_idx)
                        
                        ops.pop(idx)
                        ops.insert(idx, part1)
                        
                        insert_pos = l if l <= len(ops) else len(ops)
                        if idx < insert_pos:
                            insert_pos -= 1
                        ops.insert(insert_pos, part2)
                    break
    
    elif mu == 2:
        # Второй род: изменение порядка ЗАКАНЧИВАЯ машиной m_idx
        # Применяем изменения на машинах 0, 1, ..., m_idx
        
        for machine_idx in range(0, m_idx + 1):
            ops = new_sched.machines[machine_idx]
            
            # Находим работу с тем же job_id (или берем с позиции k на машине m_idx)
            if machine_idx == m_idx and k < len(ops):
                target = ops[k]
            else:
                # Ищем работу с тем же ID
                target = None
                for pj in ops:
                    if pj.job_id == target_pj.job_id:
                        target = pj
                        break
                if target is None:
                    continue
            
            # Разрезаем и перемещаем
            part1_dur = target.duration // 2
            part2_dur = target.duration - part1_dur
            
            if part1_dur > 0 and part2_dur > 0 and machine_idx == m_idx:
                # Только на машине m_idx разрезаем
                part1 = PseudoJob(target.job_id, part1_dur, target.machine_start)
                part2 = PseudoJob(target.job_id, part2_dur, m_idx)
                
                idx = ops.index(target)
                ops.pop(idx)
                ops.insert(idx, part1)
                
                insert_pos = l if l <= len(ops) else len(ops)
                if idx < insert_pos:
                    insert_pos -= 1
                ops.insert(insert_pos, part2)
    
    return new_sched


def apply_operator_simple(schedule: Schedule, mu: int, k: int, l: int, m_idx: int) -> Schedule:
    """
    Упрощенная версия оператора для случая без разбиения работ.
    Просто меняет порядок работ на указанных машинах.
    """
    if k < 0 or l < 0 or m_idx < 0 or m_idx >= schedule.m:
        return schedule.copy()
    
    if len(schedule.machines[m_idx]) <= k:
        return schedule.copy()
    
    new_sched = schedule.copy()
    ops = new_sched.machines[m_idx]
    
    # Извлекаем работу
    target_pj = ops.pop(k)
    
    # Корректируем позицию вставки
    insert_pos = l
    if k < l:
        insert_pos = l - 1
    insert_pos = max(0, min(insert_pos, len(ops)))
    
    # Вставляем на новую позицию
    ops.insert(insert_pos, target_pj)
    
    if mu == 1:
        # Применяем на последующих машинах
        for machine_idx in range(m_idx + 1, new_sched.m):
            other_ops = new_sched.machines[machine_idx]
            # Находим работу с тем же ID
            for idx, pj in enumerate(other_ops):
                if pj.job_id == target_pj.job_id:
                    other_ops.pop(idx)
                    insert_p = min(insert_pos, len(other_ops))
                    other_ops.insert(insert_p, pj)
                    break
    
    elif mu == 2:
        # Применяем на предыдущих машинах
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

def calculate_metric_bfs(start: Schedule, end: Schedule, max_depth: int = 3) -> int:
    """
    Оптимизированный BFS с ограничением глубины.
    """
    if start == end:
        return 0
    
    queue = deque([(start, 0)])
    visited = {hash(start)}
    
    iterations = 0
    while queue:
        iterations += 1
        if iterations % 1000 == 0:
            print(f"  ... обработано {iterations} состояний, очередь: {len(queue)}")
        
        current, dist = queue.popleft()
        if dist >= max_depth:
            continue
        
        # Генерируем ТОЛЬКО ограниченных соседей
        neighbors_generated = 0
        for m in range(min(current.m, 3)):  # Только первые 3 машины
            ops_count = len(current.machines[m])
            for k in range(min(ops_count, 5)):  # Только первые 5 позиций
                for l in range(min(ops_count + 1, 6)):  # Только первые 6 позиций
                    if k == l:
                        continue
                    
                    # Пробуем ТОЛЬКО упрощённый оператор
                    neighbor = apply_operator_simple(current, mu=1, k=k, l=l, m_idx=m)
                    h = hash(neighbor)
                    
                    if neighbor == end:
                        print(f"  Найдено за {dist + 1} шагов после {iterations} итераций")
                        return dist + 1
                    
                    if h not in visited and neighbors_generated < 50:  # Ограничение соседей
                        visited.add(h)
                        queue.append((neighbor, dist + 1))
                        neighbors_generated += 1
    
    print(f"  Не найдено в пределах глубины {max_depth} после {iterations} итераций")
    return -1

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
                   height=0.6, color=color, edgecolor='black')
            
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
    
    # Сбрасываем счетчик псевдоработ
    PseudoJob.reset_counter()
    
    # Данные из Таблицы 1 отчета (3 работы, 3 машины)
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
            pi_star.machines[m].append(PseudoJob(j_id, dur, 0))
    
    cmax_star = calculate_cmax(pi_star, job_durations)
    print(f"\n[Исходное расписание π*]")
    print(pi_star)
    print(f"Cmax = {cmax_star}")
    
    # --- Расписание π** (с изменением на машине 2) ---
    # Согласно документу: на машине 2 работа J3 выполняется перед J2
    # Это требует разбиения J2 на псевдоработы
    pi_double_star = Schedule(m_count)
    
    # Машина 1: порядок 1-2-3 (без изменений)
    for j_id in [1, 2, 3]:
        dur = job_durations[j_id][0]
        pi_double_star.machines[0].append(PseudoJob(j_id, dur, 0))
    
    # Машина 2: порядок 1-3-2 (J3 перед J2)
    # J2 разбивается на две части: J2^0 (длит. 0) и J2^1 (длит. 1)
    pi_double_star.machines[1].append(PseudoJob(1, 4, 0))  # J1
    pi_double_star.machines[1].append(PseudoJob(3, 2, 0))  # J3
    pi_double_star.machines[1].append(PseudoJob(2, 1, 0))  # J2
    
    # Машина 3: порядок 1-2-3 (но J2 разбита)
    pi_double_star.machines[2].append(PseudoJob(1, 1, 0))  # J1
    # J2 разбита на две части: 1 и 2 (или можно оставить как есть)
    pi_double_star.machines[2].append(PseudoJob(2, 1, 2))  # J2^0
    pi_double_star.machines[2].append(PseudoJob(2, 2, 2))  # J2^1 (всего 3)
    pi_double_star.machines[2].append(PseudoJob(3, 2, 0))  # J3
    
    cmax_double = calculate_cmax(pi_double_star, job_durations)
    print(f"\n[Расписание π** (изменен порядок на М2, J2 разбита)]")
    print(pi_double_star)
    print(f"Cmax = {cmax_double}")
    
    # --- Расчет метрики ---
    print(f"\n[Вычисление метрики ρΩ]")
    dist = calculate_metric_bfs(pi_star, pi_double_star, max_depth=5)
    print(f"Расстояние ρΩ(π*, π**) = {dist}")
    
    # --- Проверка неравенства треугольника ---
    # Создадим третье расписание π*** 
    pi_triple_star = Schedule(m_count)
    
    # Еще более сложное расписание с разбиением
    for m in range(m_count):
        for j_id in [1, 2, 3]:
            dur = job_durations[j_id][m]
            pi_triple_star.machines[m].append(PseudoJob(j_id, dur, 0))
    
    # Меняем на машине 3
    pi_triple_star.machines[2][0], pi_triple_star.machines[2][1] = \
        pi_triple_star.machines[2][1], pi_triple_star.machines[2][0]
    
    cmax_triple = calculate_cmax(pi_triple_star, job_durations)
    
    dist_12 = calculate_metric_bfs(pi_star, pi_double_star, max_depth=5)
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