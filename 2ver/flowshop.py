import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class CorrectScheduleVisualizer:
    """Правильная визуализация расписаний Flow Shop"""
    
    def __init__(self):
        # Добавлены цвета для нового примера из файла
        self.colors = {
            't1': '#FF6B6B',      # красный
            't2': '#4ECDC4',      # бирюзовый
            't3': '#45B7D1',      # голубой
            't4': '#FFA07A',      # светло-лососевый
            't5': '#DDA0DD',      # сливовый
            't6': '#F0E68C',      # хаки
            "t2'": '#4ECDC4',     
            "t2''": '#2FA8A0',    
            "t3'": '#45B7D1',     # t3' - как t3
            "t3''": '#287F91',    # t3'' - темнее
            'default': '#95A5A6'
        }
        self.history = []
    
    def calculate_schedule_times(self, schedule, processing_times, n_machines):
        """
        ИСПРАВЛЕННЫЙ расчет времен.
        Псевдоработы (duration == 0) больше не блокируют машины.
        """
        n_jobs = len(schedule)
        start_times = np.zeros((n_jobs, n_machines))
        completion_times = np.zeros((n_jobs, n_machines))
        
        for job_idx, job in enumerate(schedule):
            if job not in processing_times:
                continue
                
            times = processing_times[job]
            
            for machine_idx in range(n_machines):
                duration = times[machine_idx]
                
                start_time = 0
                
                # Работа может начаться не раньше, чем она закончится на предыдущей машине
                if machine_idx > 0:
                    start_time = max(start_time, completion_times[job_idx, machine_idx-1])
                
                # Работа ждет освобождения текущей машины, ТОЛЬКО если она реально на ней выполняется (duration > 0)
                if duration > 0 and job_idx > 0:
                    start_time = max(start_time, completion_times[job_idx-1, machine_idx])
                
                start_times[job_idx, machine_idx] = start_time
                completion_times[job_idx, machine_idx] = start_time + duration
        
        makespan = int(np.max(completion_times))
        return start_times, completion_times, makespan
    
    def add_step(self, step_num, schedule, processing_times, operator=None, description=""):
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
        schedule = step_data['schedule']
        processing_times = step_data['processing_times']
        start_times = step_data['start_times']
        makespan = step_data['makespan']
        operator = step_data['operator']
        
        n_machines = len(list(processing_times.values())[0])
        
        for job_idx, job in enumerate(schedule):
            if job not in processing_times:
                continue
                
            times = processing_times[job]
            color = self.colors.get(job, self.colors['default'])
            
            if highlight_jobs and job_idx in highlight_jobs:
                linewidth, edgecolor, alpha = 3, '#FF0000', 0.9
            else:
                linewidth, edgecolor, alpha = 2, 'black', 1.0
            
            for machine_idx in range(n_machines):
                duration = times[machine_idx]
                if duration == 0:
                    continue
                
                start_time = start_times[job_idx, machine_idx]
                
                rect = patches.Rectangle(
                    (start_time, machine_idx), duration, 0.7,
                    linewidth=linewidth, edgecolor=edgecolor,
                    facecolor=color, alpha=alpha, zorder=3
                )
                ax.add_patch(rect)
                
                # ИСПРАВЛЕНИЕ НАСЛОЕНИЯ ТЕКСТА
                if duration >= 2.0:  # Увеличен порог
                    ax.text(
                        start_time + duration/2,
                        machine_idx + 0.35,
                        job,
                        ha='center', va='center',
                        fontsize=8,           # Уменьшен шрифт
                        fontweight='bold',
                        color='white',
                        clip_on=True,         # Обрезка по границам графика
                        zorder=4
                    )
        
        ax.set_ylim(-0.5, n_machines + 0.5)
        ax.set_xlim(0, makespan + 5)
        ax.set_yticks(range(n_machines))
        ax.set_yticklabels([f'M{i+1}' for i in range(n_machines)], fontsize=11, fontweight='bold')
        ax.set_xlabel('Время', fontsize=11, fontweight='bold')
        
        title = f"{step_data['description']}\nДлина расписания: {makespan}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_axisbelow(True)
        
        if operator:
            # Универсальный вывод оператора для разных размерностей
            op_text = f'Ω({", ".join(map(str, operator))})'
            ax.text(0.02, 0.98, op_text, transform=ax.transAxes, 
                    fontsize=9, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8),
                    verticalalignment='top')
    
    def visualize_all_steps(self, title, save_path=None):
        n_steps = len(self.history)
        n_cols = 2
        n_rows = (n_steps + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        
        if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
        elif n_rows == 1: axes = axes.reshape(1, -1)
        elif n_cols == 1: axes = axes.reshape(-1, 1)
        
        for idx, step_data in enumerate(self.history):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            highlight = None
            if step_data['operator']:
                k = step_data['operator'][1]
                highlight = [k-1] if k > 0 else None
                
            self.plot_gantt_chart(ax, step_data, highlight)
            
        for idx in range(n_steps, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            fig.delaxes(axes[row, col])
            
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Изображение сохранено: {save_path}")
        plt.close()


# ============================================================================
# ПРИМЕР 1: Из исходного кода (4 работы, 4 машины)
# ============================================================================
def create_complex_example():
    viz = CorrectScheduleVisualizer()
    base_times = {
        't1': [2, 3, 4, 2], 't2': [5, 2, 3, 4],
        't3': [3, 4, 2, 3], 't4': [4, 3, 5, 2]
    }
    
    viz.add_step(1, ['t1', 't2', 't3', 't4'], base_times.copy(), 
                 None, "ШАГ 1: Исходное расписание π₁")
    
    t1 = base_times.copy()
    t1["t2'"] = [5, 0, 0, 0]
    t1["t2''"] = [0, 2, 3, 4]
    
    viz.add_step(2, ['t1', "t2'", 't3', 't4', "t2''"], t1.copy(), 
                 (1, 2, 5, 2), "ШАГ 2: Разбиение работы t2 на псевдоработы t2' и t2''")
    
    viz.add_step(3, ["t2'", 't1', 't3', 't4', "t2''"], t1.copy(), 
                 (1, 1, 2, 4), "ШАГ 3: Перемещение t1")
    
    viz.add_step(4, ["t2'", 't3', 't4', "t2''", 't1'], t1.copy(), 
                 (1, 2, 5, 4), "ШАГ 4: Перемещение t1 в конец")
    
    viz.add_step(5, ["t2'", "t2''", 't3', 't4', 't1'], t1.copy(), 
                 (1, 2, 4, 4), "ШАГ 5: Сдвиг t3 и t4")
    
    viz.add_step(6, ["t2'", "t2''", 't1', 't3', 't4'], t1.copy(), 
                 (1, 5, 3, 4), "ШАГ 6: Перемещение t1 на позицию 3")
    
    viz.add_step(7, ['t2', 't1', 't3', 't4'], base_times.copy(), 
                 None, "ШАГ 7: Слияние соседних t2' и t2'' обратно в целую t2")
    
    viz.add_step(8, ['t2', 't1', 't4', 't3'], base_times.copy(), 
                 (1, 3, 4, 4), "ШАГ 8: ЦЕЛЕВОЕ РАСПИСАНИЕ π₂ (Финальная перестановка)")
    
    return viz

# ============================================================================
# ПРИМЕР 2: Из файла «Нир 3.0.docx» (6 работ, 5 машин) - ПОЛНАЯ ВЕРСИЯ
# ============================================================================
def create_doc_example():
    viz = CorrectScheduleVisualizer()
    base_times = {
        't1': [3, 15, 2, 4, 5],
        't2': [7, 8, 6, 5, 9],
        't3': [4, 12, 20, 3, 6],
        't4': [9, 3, 8, 15, 4],
        't5': [5, 6, 4, 7, 18],
        't6': [6, 10, 5, 8, 7]
    }
    
    # ШАГ 1
    schedule_0 = ['t1', 't2', 't3', 't4', 't5', 't6']
    viz.add_step(1, schedule_0, base_times.copy(), 
                None, "ШАГ 1. Исходное расписание π1 (Длина: 113)")
    
    # ШАГ 2
    schedule_1 = ['t1', 't2', "t3'", 't4', 't5', 't6', "t3''"]
    times_1 = base_times.copy()
    times_1["t3'"] = [4, 12, 0, 0, 0]   # Операции на M1, M2
    times_1["t3''"] = [0, 0, 20, 3, 6]  # Операции на M3, M4, M5
    del times_1['t3']
    
    viz.add_step(2, schedule_1, times_1.copy(),
                (1, 3, 7, 3), "ШАГ 2. Разбиение t3 на t3' и t3''. Перенос t3'' в конец (Длина: 102)")
    
    # ШАГ 3 (Финальный)
    # Перемещаем t1 после t4 и сливаем t3' и t3'' обратно
    schedule_2 = ['t3', 't2', 't1', 't4', 't5', 't6']
    viz.add_step(3, schedule_2, base_times.copy(),
                (1, 1, 4, 5), "ШАГ 3. Сдвиг t1 и слияние t3' + t3'' обратно в t3 (Длина: 99)")
    
    return viz

if __name__ == "__main__":
    print("Генерация Примера 1 (4x4)...")
    viz1 = create_complex_example()
    viz1.visualize_all_steps(
        title='ПРЕОБРАЗОВАНИЕ РАСПИСАНИЯ (4 работы, 4 машины)',
        save_path='complex_schedule_transformation.png'
    )
    
    print("Генерация Примера 2 из файла (6x5)...")
    viz2 = create_doc_example()
    viz2.visualize_all_steps(
        title='АНАЛИЗ И УСТРАНЕНИЕ УЗКИХ МЕСТ (6 работ, 5 машин - Нир 3.0)',
        save_path='doc_example_transformation.png'
    )
    
    print("\n✅ ГОТОВО! Сохранены два файла: complex_schedule_transformation.png и doc_example_transformation.png")