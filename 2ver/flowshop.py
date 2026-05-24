import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Patch

class CorrectScheduleVisualizer:
    """Правильная визуализация расписаний Flow Shop"""
    
    def __init__(self):
        self.colors = {
            't1': '#FF6B6B',      # красный
            't2': '#4ECDC4',      # бирюзовый
            't3': '#45B7D1',      # голубой
            't4': '#FFA07A',      # светло-лососевый
            "t2'": '#4ECDC4',     # t2' - светлее
            "t2''": '#2FA8A0',    # t2'' - темнее
            'default': '#95A5A6'
        }
        self.history = []
    
    def calculate_schedule_times(self, schedule, processing_times, n_machines):
        """
        ПРАВИЛЬНО рассчитывает времена начала и окончания операций
        для задачи Flow Shop
        """
        n_jobs = len(schedule)
        
        # Матрицы времен начала и окончания
        start_times = np.zeros((n_jobs, n_machines))
        completion_times = np.zeros((n_jobs, n_machines))
        
        for job_idx, job in enumerate(schedule):
            if job not in processing_times:
                continue
                
            times = processing_times[job]
            
            for machine_idx in range(n_machines):
                duration = times[machine_idx]
                
                if duration == 0:
                    if job_idx == 0:
                        completion_times[job_idx, machine_idx] = 0
                    else:
                        completion_times[job_idx, machine_idx] = completion_times[job_idx-1, machine_idx]
                    start_times[job_idx, machine_idx] = completion_times[job_idx, machine_idx]
                    continue
                
                # Время начала = max(
                #   окончание предыдущей работы на этой машине,
                #   окончание этой работы на предыдущей машине
                # )
                start_time = 0
                
                if job_idx > 0:
                    start_time = max(start_time, completion_times[job_idx-1, machine_idx])
                
                if machine_idx > 0:
                    start_time = max(start_time, completion_times[job_idx, machine_idx-1])
                
                start_times[job_idx, machine_idx] = start_time
                completion_times[job_idx, machine_idx] = start_time + duration
        
        makespan = int(np.max(completion_times))
        return start_times, completion_times, makespan
    
    def add_step(self, step_num, schedule, processing_times, operator=None, 
                 description=""):
        """Добавляет шаг в историю"""
        n_machines = len(list(processing_times.values())[0])
        start_times, completion_times, makespan = self.calculate_schedule_times(
            schedule, processing_times, n_machines
        )
        
        self.history.append({
            'step': step_num,
            'schedule': schedule.copy(),
            'processing_times': processing_times.copy(),
            'operator': operator,
            'description': description,
            'start_times': start_times,
            'completion_times': completion_times,
            'makespan': makespan
        })
    
    def plot_gantt_chart(self, ax, step_data, highlight_jobs=None):
        """Рисует правильную диаграмму Ганта"""
        schedule = step_data['schedule']
        processing_times = step_data['processing_times']
        start_times = step_data['start_times']
        completion_times = step_data['completion_times']
        makespan = step_data['makespan']
        operator = step_data['operator']
        
        n_machines = len(list(processing_times.values())[0])
        n_jobs = len(schedule)
        
        # Рисуем операции
        for job_idx, job in enumerate(schedule):
            if job not in processing_times:
                continue
                
            times = processing_times[job]
            color = self.colors.get(job, self.colors['default'])
            
            # Выделение изменяемых работ
            if highlight_jobs and job_idx in highlight_jobs:
                linewidth = 3
                edgecolor = '#FF0000'
                alpha = 0.9
            else:
                linewidth = 2
                edgecolor = 'black'
                alpha = 1.0
            
            for machine_idx in range(n_machines):
                duration = times[machine_idx]
                if duration == 0:
                    continue
                
                start_time = start_times[job_idx, machine_idx]
                
                # Прямоугольник операции
                rect = patches.Rectangle(
                    (start_time, machine_idx),
                    duration,
                    0.7,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor=color,
                    alpha=alpha,
                    zorder=3
                )
                ax.add_patch(rect)
                
                # Текст с названием работы
                if duration >= 1.5:
                    ax.text(
                        start_time + duration/2,
                        machine_idx + 0.35,
                        job,
                        ha='center',
                        va='center',
                        fontsize=9,
                        fontweight='bold',
                        color='white',
                        zorder=4
                    )
        
        # Настройка осей
        ax.set_ylim(-0.5, n_machines + 0.5)
        ax.set_xlim(0, makespan + 5)
        
        ax.set_yticks(range(n_machines))
        ax.set_yticklabels([f'M{i+1}' for i in range(n_machines)], 
                          fontsize=11, fontweight='bold')
        ax.set_xlabel('Время', fontsize=11, fontweight='bold')
        
        # Заголовок
        title = f"{step_data['description']}\nДлина расписания: {makespan}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_axisbelow(True)
        
        # Информация об операторе
        if operator:
            mu, k, l, m = operator
            op_text = f'Ω(μ={mu}, k={k}, l={l}, m={m})'
            ax.text(0.02, 0.98, op_text, transform=ax.transAxes, 
                   fontsize=9, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8),
                   verticalalignment='top')
    
    def visualize_all_steps(self, save_path=None):
        """Визуализирует все шаги"""
        n_steps = len(self.history)
        n_cols = 2
        n_rows = (n_steps + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, step_data in enumerate(self.history):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            highlight = None
            if step_data['operator']:
                k = step_data['operator'][1]
                highlight = [k-1] if k > 0 else None
            
            self.plot_gantt_chart(ax, step_data, highlight)
        
        # Удаляем пустые подграфики
        for idx in range(n_steps, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        # Общий заголовок
        total_makespan = self.history[-1]['makespan']
        fig.suptitle(
            f'ПРЕОБРАЗОВАНИЕ РАСПИСАНИЯ (4 работы, 4 машины)\n'
            f'Всего шагов: {n_steps} | Итоговая длина: {total_makespan}',
            fontsize=16, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Изображение сохранено: {save_path}")
        
        plt.show()
    
    def print_details(self):
        """Выводит подробную информацию"""
        print("=" * 80)
        print("ДЕТАЛЬНОЕ ОПИСАНИЕ ПРЕОБРАЗОВАНИЯ")
        print("=" * 80)
        
        for step_data in self.history:
            print(f"\n{'='*80}")
            print(f"ШАГ {step_data['step']}: {step_data['description']}")
            print(f"{'='*80}")
            print(f"Расписание: {' → '.join(step_data['schedule'])}")
            print(f"Длина: {step_data['makespan']}")
            
            if step_data['operator']:
                mu, k, l, m = step_data['operator']
                print(f"Оператор: Ω(μ={mu}, k={k}, l={l}, m={m})")


# ============================================================================
# СЛОЖНЫЙ ПРИМЕР: 4 работы, 4 машины
# ============================================================================

def create_complex_example():
    """
    Пример с 4 работами и 4 машинами
    Преобразование: t1 t2 t3 t4 → t2 t1 t4 t3
    """
    viz = CorrectScheduleVisualizer()
    n_machines = 4
    
    # Матрица времен (4 работы × 4 машины)
    base_processing_times = {
        't1': [2, 3, 4, 2],
        't2': [5, 2, 3, 4],
        't3': [3, 4, 2, 3],
        't4': [4, 3, 5, 2]
    }
    
    # ШАГ 0: Исходное расписание
    schedule_0 = ['t1', 't2', 't3', 't4']
    viz.add_step(0, schedule_0, base_processing_times.copy(), 
                None, "Исходное расписание π₁")
    
    # ШАГ 1: Разбиение t2 на псевдоработы
    schedule_1 = ['t1', "t2'", 't3', 't4', "t2''"]
    processing_times_1 = {
        't1': [2, 3, 4, 2],
        "t2'": [5, 0, 0, 0],
        't3': [3, 4, 2, 3],
        't4': [4, 3, 5, 2],
        "t2''": [0, 2, 3, 4]
    }
    viz.add_step(1, schedule_1, processing_times_1.copy(),
                (1, 2, 5, 2), "Разбиение t2 → t2' + t2''")
    
    # ШАГ 2: Перемещение t1 после t2'
    schedule_2 = ["t2'", 't1', 't3', 't4', "t2''"]
    viz.add_step(2, schedule_2, processing_times_1.copy(),
                (1, 1, 2, 4), "Перемещение t1 после t2'")
    
    # ШАГ 3: Перемещение t1 дальше (после t2'')
    schedule_3 = ["t2'", 't3', 't4', "t2''", 't1']
    viz.add_step(3, schedule_3, processing_times_1.copy(),
                (1, 2, 5, 4), "Перемещение t1 после t2''")
    
    # ШАГ 4: Перемещение t3 и t4 для сближения t2' и t2''
    schedule_4 = ["t2'", "t2''", 't3', 't4', 't1']
    viz.add_step(4, schedule_4, processing_times_1.copy(),
                (1, 2, 4, 4), "t2' и t2'' стали соседними")
    
    # ШАГ 5: Перемещение t1 на позицию 3
    schedule_5 = ["t2'", "t2''", 't1', 't3', 't4']
    viz.add_step(5, schedule_5, processing_times_1.copy(),
                (1, 5, 3, 4), "Перемещение t1 на позицию 3")
    
    # ШАГ 6: Слияние t2' и t2''
    schedule_6 = ['t2', 't1', 't3', 't4']
    viz.add_step(6, schedule_6, base_processing_times.copy(),
                None, "Слияние t2' + t2'' → t2")
    
    # ШАГ 7: Перемещение t3 и t4 (финальная настройка)
    schedule_7 = ['t2', 't1', 't4', 't3']
    viz.add_step(7, schedule_7, base_processing_times.copy(),
                (1, 3, 4, 4), "ЦЕЛЕВОЕ РАСПИСАНИЕ π₂!")
    
    return viz


if __name__ == "__main__":
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ СЛОЖНОГО ПРИМЕРА (4 работы, 4 машины)")
    print("=" * 80)
    
    viz = create_complex_example()
    viz.print_details()
    
    print("\n\nГенерация визуализации...")
    viz.visualize_all_steps(save_path='complex_schedule_transformation.png')
    
    print("\n✅ ГОТОВО!")