import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from scipy.signal import find_peaks, lfilter, freqz
import librosa  # 用于LPC分析
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os
import traceback
import sys

# 设置中文字体支持
def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            print("警告: 无法正确设置中文字体，图表标签可能显示异常")
            return False

# 调用字体设置
set_chinese_font()

class DynamicFormantAnalyzer:
    def __init__(self, master):
        self.master = master
        master.title("动态语音共振峰分析")
        master.geometry("1200x800")
        
        # 创建主框架
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建绘图区域
        self.create_plot_area()
        
        # 初始化变量
        self.rate = None
        self.data = None
        self.filename = None
        self.formants = []
        self.times = []
        self.anim = None
        
        # 人类语音共振峰参考范围 (F1-F3)
        self.human_formant_ranges = {
            'F1': (100, 1000),
            'F2': (800, 2500),
            'F3': (2000, 3500)
        }
        
        # 设置初始状态
        self.update_status("就绪，请选择语音文件")
    
    def create_control_panel(self):
        # 控制面板框架
        control_frame = tk.LabelFrame(self.main_frame, text="控制面板")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 文件选择按钮
        file_frame = tk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_open = tk.Button(file_frame, text="选择语音文件", command=self.load_audio)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(file_frame, text="未选择文件", width=60, anchor=tk.W)
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        # 按钮面板
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_analyze = tk.Button(btn_frame, text="开始分析", command=self.analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = tk.Button(btn_frame, text="播放动画", command=self.toggle_animation, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = tk.Button(btn_frame, text="保存结果", command=self.save_results, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        # 参数设置区域
        param_frame = tk.LabelFrame(control_frame, text="分析参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 分析方法选择
        method_frame = tk.Frame(param_frame)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(method_frame, text="分析方法:").pack(side=tk.LEFT, padx=5)
        self.method_var = tk.StringVar(value="cepstrum")
        tk.Radiobutton(method_frame, text="倒谱法", variable=self.method_var, 
                      value="cepstrum").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(method_frame, text="LPC法", variable=self.method_var, 
                      value="lpc").pack(side=tk.LEFT, padx=5)
        
        # 第一行参数
        param_row1 = tk.Frame(param_frame)
        param_row1.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(param_row1, text="帧长(ms):").pack(side=tk.LEFT, padx=5)
        self.frame_length = tk.IntVar(value=30)
        tk.Entry(param_row1, textvariable=self.frame_length, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_row1, text="帧移(ms):").pack(side=tk.LEFT, padx=5)
        self.frame_shift = tk.IntVar(value=10)
        tk.Entry(param_row1, textvariable=self.frame_shift, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_row1, text="共振峰数量:").pack(side=tk.LEFT, padx=5)
        self.n_formants = tk.IntVar(value=3)
        tk.Entry(param_row1, textvariable=self.n_formants, width=5).pack(side=tk.LEFT, padx=5)
        
        # 第二行参数
        param_row2 = tk.Frame(param_frame)
        param_row2.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(param_row2, text="最小频率(Hz):").pack(side=tk.LEFT, padx=5)
        self.min_freq = tk.IntVar(value=80)
        tk.Entry(param_row2, textvariable=self.min_freq, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_row2, text="最大频率(Hz):").pack(side=tk.LEFT, padx=5)
        self.max_freq = tk.IntVar(value=5000)
        tk.Entry(param_row2, textvariable=self.max_freq, width=5).pack(side=tk.LEFT, padx=5)
        
        # 第三行参数
        param_row3 = tk.Frame(param_frame)
        param_row3.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(param_row3, text="峰值高度阈值:").pack(side=tk.LEFT, padx=5)
        self.peak_height = tk.DoubleVar(value=0.01)
        tk.Entry(param_row3, textvariable=self.peak_height, width=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_row3, text="峰值距离:").pack(side=tk.LEFT, padx=5)
        self.peak_distance = tk.IntVar(value=10)
        tk.Entry(param_row3, textvariable=self.peak_distance, width=5).pack(side=tk.LEFT, padx=5)
        
        # 状态栏和进度条
        status_frame = tk.Frame(control_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 创建可滚动的状态标签
        status_container = tk.Frame(status_frame)
        status_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 使用Text小部件替代Label，以便显示更多错误信息
        self.status_text = tk.Text(status_container, height=3, width=60, wrap=tk.WORD, bg=self.master.cget('bg'), 
                                 relief=tk.FLAT, borderwidth=0, highlightthickness=0)
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_text.insert(tk.END, "就绪")
        self.status_text.config(state=tk.DISABLED)  # 设置为只读
    
    def update_status(self, message):
        """更新状态栏，显示更多错误信息"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.config(state=tk.DISABLED)
        self.master.update()
    
    def create_plot_area(self):
        # 创建绘图区域框架
        plot_frame = tk.Frame(self.main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化绘图区域
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(311)  # 波形图
        self.ax2 = self.fig.add_subplot(312)  # 共振峰轨迹
        self.ax3 = self.fig.add_subplot(313)  # 声谱图
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_audio(self):
        filetypes = [("WAV files", "*.wav"), ("All files", "*.*")]
        
        try:
            filename = filedialog.askopenfilename(
                title="选择语音文件",
                filetypes=filetypes
            )
            if filename:
                self.filename = filename
                self.rate, self.data = wav.read(filename)
                
                # 处理音频数据
                if self.data.ndim > 1:
                    self.data = self.data[:, 0]  # 取单声道
                
                # 归一化处理
                max_val = np.max(np.abs(self.data))
                if max_val > 0:
                    self.data = self.data.astype(np.float32) / max_val
                
                # 添加预加重滤波器 (增强高频分量)
                pre_emphasis = 0.97
                self.data = np.append(self.data[0], self.data[1:] - pre_emphasis * self.data[:-1])
                
                # 更新状态和文件标签
                duration = len(self.data) / self.rate
                self.file_label.config(text=f"已加载: {os.path.basename(filename)} (时长: {duration:.2f}秒, 采样率: {self.rate}Hz)")
                self.btn_analyze.config(state=tk.NORMAL)
                
                # 绘制初始波形图
                self.ax1.clear()
                time_axis = np.arange(len(self.data)) / self.rate
                self.ax1.plot(time_axis, self.data, 'b')
                self.ax1.set_title('语音波形')
                self.ax1.set_ylabel('振幅')
                self.ax1.set_xlim(0, time_axis[-1])
                self.ax1.grid(True, alpha=0.3)
                self.canvas.draw()
        except Exception as e:
            self.update_status(f"错误: {str(e)}\n{traceback.format_exc()}")
    
    def analyze(self):
        if self.data is None:
            self.update_status("错误: 没有加载音频文件")
            return
        
        # 禁用分析按钮
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_play.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        
        # 获取参数
        frame_length_ms = self.frame_length.get()
        frame_shift_ms = self.frame_shift.get()
        n_formants = self.n_formants.get()
        min_freq = self.min_freq.get()
        max_freq = self.max_freq.get()
        peak_height = self.peak_height.get()
        peak_distance = self.peak_distance.get()
        method = self.method_var.get()
        
        # 计算帧长和帧移（样本数）
        frame_len = int(self.rate * frame_length_ms / 1000)
        frame_shift = int(self.rate * frame_shift_ms / 1000)
        
        # 初始化结果存储
        self.formants = []  # 存储每帧的共振峰频率
        self.times = []     # 存储每帧的时间点（中心时间）
        
        # 创建汉宁窗
        window = np.hanning(frame_len)
        
        # 初始化进度条
        num_frames = (len(self.data) - frame_len) // frame_shift + 1
        self.progress["maximum"] = num_frames
        self.progress["value"] = 0
        
        # 开始分析
        start_time = time.time()
        frames_with_formants = 0
        
        for i in range(0, len(self.data) - frame_len, frame_shift):
            # 提取当前帧
            frame = self.data[i:i+frame_len]
            
            # 加窗
            windowed = frame * window
            
            if method == "cepstrum":
                # 计算倒谱
                spectrum = np.fft.rfft(windowed)
                log_spectrum = np.log(np.abs(spectrum) + 1e-10)
                cepstrum = np.fft.irfft(log_spectrum).real
                
                # 设置有效的倒谱分析范围
                min_quef = int(self.rate / max_freq)  # 最高频率对应的倒频率
                max_quef = int(self.rate / min_freq)   # 最低频率对应的倒频率
                
                # 确保范围有效
                low_bound = max(2, min_quef)
                high_bound = min(len(cepstrum)//2, max_quef)
                
                if low_bound < high_bound:
                    valid_range = slice(low_bound, high_bound)
                    
                    # 提取有效倒谱段
                    valid_cepstrum = cepstrum[valid_range]
                    
                    # 使用峰值检测算法
                    peaks, properties = find_peaks(
                        valid_cepstrum, 
                        height=peak_height,
                        distance=peak_distance,
                        prominence=0.1,
                        width=1
                    )
                    
                    # 存储当前帧的共振峰
                    frame_formants = []
                    if len(peaks) > 0:
                        # 按峰值高度排序（最高峰优先）
                        peak_heights = properties['peak_heights']
                        sorted_indices = np.argsort(peak_heights)[::-1]  # 从高到低排序
                        
                        # 选择最高的n_formants个峰值
                        if len(sorted_indices) > n_formants:
                            selected_peaks = peaks[sorted_indices[:n_formants]]
                        else:
                            selected_peaks = peaks[sorted_indices]
                        
                        # 计算共振峰频率
                        formant_quef = (selected_peaks + valid_range.start) / self.rate
                        formant_freqs = 1 / formant_quef
                        frame_formants = np.sort(formant_freqs)
                        frames_with_formants += 1
                else:
                    frame_formants = []
            elif method == "lpc":
                # 使用LPC方法提取共振峰
                frame_formants = self.extract_formants_lpc(
                    windowed, self.rate, n_formants,
                    min_freq, max_freq
                )
                if frame_formants:
                    frames_with_formants += 1
            
            self.formants.append(frame_formants)
            self.times.append(i / self.rate + frame_length_ms / 2000)  # 帧中心时间
            
            # 更新进度
            current_frame = i // frame_shift
            self.progress["value"] = current_frame
            if current_frame % 10 == 0:  # 每10帧更新一次状态
                elapsed = time.time() - start_time
                frames_processed = current_frame + 1
                if frames_processed > 0:
                    time_per_frame = elapsed / frames_processed
                    remaining = time_per_frame * (num_frames - frames_processed)
                    self.update_status(f"分析中... {frames_processed}/{num_frames} | 方法: {method} | 预计剩余: {remaining:.1f}秒")
        
        # 分析完成
        elapsed = time.time() - start_time
        self.update_status(f"分析完成! 共分析 {len(self.formants)} 帧 | 检测到共振峰: {frames_with_formants}帧 | 耗时: {elapsed:.2f}秒")
        self.btn_play.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        
        # 绘制结果
        self.plot_results()
    
    def extract_formants_lpc(self, frame, rate, n_formants, min_freq, max_freq):
        """使用LPC方法提取共振峰"""
        try:
            # LPC阶数 (一般取采样率/1000 + 2-4)
            order = int(rate / 1000) + 3
            
            # 计算LPC系数
            a = librosa.lpc(frame, order)
            
            # 计算频率响应
            w, h = freqz(1, a, worN=2048)
            freqs = w * rate / (2 * np.pi)
            
            # 寻找频谱峰值
            peaks, _ = find_peaks(np.abs(h), height=0.01, distance=10)
            
            # 过滤频率范围
            valid_peaks = [p for p in peaks if min_freq <= freqs[p] <= max_freq]
            
            # 按幅度排序并选择前n_formants个
            if valid_peaks:
                peak_amps = np.abs(h)[valid_peaks]
                sorted_indices = np.argsort(peak_amps)[::-1]
                selected_peaks = sorted_indices[:min(n_formants, len(valid_peaks))]
                formant_freqs = [freqs[valid_peaks[i]] for i in selected_peaks]
                return np.sort(formant_freqs)
        except Exception as e:
            self.update_status(f"LPC分析错误: {str(e)}")
            return []
        return []
    
    def plot_results(self):
        if not self.formants:
            self.update_status("错误: 没有分析结果可绘制")
            return
        
        try:
            # 清除之前的绘图
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # 绘制波形图
            time_axis = np.arange(len(self.data)) / self.rate
            self.ax1.plot(time_axis, self.data, 'b')
            self.ax1.set_title('语音波形')
            self.ax1.set_ylabel('振幅')
            self.ax1.set_xlim(0, time_axis[-1])
            self.ax1.grid(True, alpha=0.3)
            
            # 准备共振峰数据
            f1 = [f[0] if len(f) > 0 else np.nan for f in self.formants]
            f2 = [f[1] if len(f) > 1 else np.nan for f in self.formants]
            f3 = [f[2] if len(f) > 2 else np.nan for f in self.formants]
            
            # 修复插值函数 - 确保使用整数索引
            def interpolate_missing(values):
                # 确保值是浮点数数组
                values = np.array(values, dtype=float)
                indices = np.arange(len(values))
                valid = ~np.isnan(values)
                
                # 检查是否有足够的数据进行插值
                if np.sum(valid) > 1:
                    # 确保索引是整数
                    valid_indices = indices[valid].astype(int)
                    valid_values = values[valid]
                    
                    # 确保索引是单调递增的
                    sorted_idx = np.argsort(valid_indices)
                    sorted_indices = valid_indices[sorted_idx]
                    sorted_values = valid_values[sorted_idx]
                    
                    # 使用线性插值
                    return np.interp(indices, sorted_indices, sorted_values)
                return values
            
            # 应用插值
            f1 = interpolate_missing(f1)
            f2 = interpolate_missing(f2)
            f3 = interpolate_missing(f3)
            
            # 中值滤波平滑数据
            def median_filter(data, window_size=5):
                filtered = []
                for i in range(len(data)):
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(data), i + window_size + 1)
                    window_data = data[start_idx:end_idx]
                    
                    # 过滤掉NaN值
                    non_nan_data = [x for x in window_data if not np.isnan(x)]
                    if non_nan_data:
                        filtered.append(np.median(non_nan_data))
                    else:
                        filtered.append(np.nan)
                return filtered
            
            # 应用滤波
            f1 = median_filter(f1)
            f2 = median_filter(f2)
            f3 = median_filter(f3)
            
            # 绘制共振峰轨迹
            self.ax2.plot(self.times, f1, 'r-', label='F1')
            self.ax2.plot(self.times, f2, 'g-', label='F2')
            self.ax2.plot(self.times, f3, 'b-', label='F3')
            self.ax2.set_title('共振峰轨迹')
            self.ax2.set_ylabel('频率 (Hz)')
            self.ax2.set_xlim(0, self.times[-1])
            self.ax2.set_ylim(0, self.max_freq.get())
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # 绘制人类共振峰参考范围
            for i, (formant, (low, high)) in enumerate(self.human_formant_ranges.items()):
                color = ['r', 'g', 'b'][i]
                self.ax2.axhspan(low, high, alpha=0.1, color=color)
                self.ax2.text(self.times[-1]*0.95, (low+high)/2, 
                             f'{formant}范围', color=color, 
                             ha='right', va='center')
            
            # 绘制声谱图
            # 计算声谱图
            frame_length_ms = self.frame_length.get()
            frame_len = int(self.rate * frame_length_ms / 1000)
            n_fft = 1024
            n_frames = len(self.times)
            spec = np.zeros((n_fft//2 + 1, n_frames))
            
            # 计算频率轴
            freqs = np.fft.rfftfreq(n_fft, 1.0/self.rate)
            
            for i, t in enumerate(self.times):
                # 确保索引是整数
                center_index = int(t * self.rate)
                start_idx = center_index - frame_len // 2
                end_idx = center_index + frame_len // 2
                
                # 边界检查
                if start_idx < 0:
                    start_idx = 0
                    end_idx = frame_len
                if end_idx > len(self.data):
                    end_idx = len(self.data)
                    start_idx = end_idx - frame_len
                if start_idx < 0:  # 再次检查，确保安全
                    start_idx = 0
                
                # 提取帧数据
                frame = self.data[start_idx:end_idx]
                if len(frame) < frame_len:
                    # 填充零以匹配帧长
                    frame = np.pad(frame, (0, frame_len - len(frame)), 'constant')
                
                # 加窗和FFT
                windowed = frame * np.hanning(frame_len)
                spectrum = np.abs(np.fft.rfft(windowed, n_fft))
                spec[:, i] = 20 * np.log10(spectrum + 1e-10)
            
            # 绘制声谱图
            max_freq = self.max_freq.get()
            freq_mask = (freqs <= max_freq)
            freqs_filtered = freqs[freq_mask]
            spec_filtered = spec[freq_mask, :]
            
            # 创建时间轴
            times = np.array(self.times)
            
            # 绘制声谱图
            if spec_filtered.size > 0:  # 确保有数据可绘制
                im = self.ax3.imshow(
                    spec_filtered, 
                    aspect='auto', 
                    origin='lower', 
                    extent=[times.min(), times.max(), freqs_filtered.min(), freqs_filtered.max()],
                    cmap='viridis', 
                    vmin=-50, 
                    vmax=0
                )
                self.ax3.set_title('声谱图')
                self.ax3.set_xlabel('时间 (秒)')
                self.ax3.set_ylabel('频率 (Hz)')
                self.ax3.set_ylim(0, max_freq)
                
                # 在声谱图上叠加共振峰轨迹
                self.ax3.plot(self.times, f1, 'r-', linewidth=1.5, label='F1')
                self.ax3.plot(self.times, f2, 'g-', linewidth=1.5, label='F2')
                self.ax3.plot(self.times, f3, 'b-', linewidth=1.5, label='F3')
                self.ax3.legend()
                
                # 添加颜色条
                self.fig.colorbar(im, ax=self.ax3, label='强度 (dB)')
            
            # 调整布局
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            error_trace = traceback.format_exc()
            # 使用消息框显示完整错误信息
            messagebox.showerror("绘图错误", f"发生错误:\n{error_trace}")
            # 在状态栏也显示错误摘要
            self.update_status(f"绘图错误: {str(e)}")
    
    def toggle_animation(self):
        if not hasattr(self, 'anim') or not self.anim.running:
            self.start_animation()
            self.btn_play.config(text="停止动画")
        else:
            self.stop_animation()
            self.btn_play.config(text="播放动画")
    
    def start_animation(self):
        if not self.formants:
            self.update_status("错误: 没有分析结果可动画化")
            return
        
        # 创建动画函数
        def animate(i):
            # 更新垂直线位置
            self.time_line1.set_xdata([self.times[i], self.times[i]])
            self.time_line2.set_xdata([self.times[i], self.times[i]])
            self.time_line3.set_xdata([self.times[i], self.times[i]])
            
            # 更新标记点
            if i < len(self.formants) and len(self.formants[i]) > 0:
                self.marker1.set_data([self.times[i]], [self.formants[i][0]])
                if len(self.formants[i]) > 1:
                    self.marker2.set_data([self.times[i]], [self.formants[i][1]])
                if len(self.formants[i]) > 2:
                    self.marker3.set_data([self.times[i]], [self.formants[i][2]])
            
            return self.time_line1, self.time_line2, self.time_line3, self.marker1, self.marker2, self.marker3
        
        # 添加时间标记线
        self.time_line1 = self.ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        self.time_line2 = self.ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        self.time_line3 = self.ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # 添加标记点
        self.marker1, = self.ax2.plot([], [], 'ro', markersize=8, alpha=0.7)
        self.marker2, = self.ax2.plot([], [], 'go', markersize=8, alpha=0.7)
        self.marker3, = self.ax2.plot([], [], 'bo', markersize=8, alpha=0.7)
        
        # 创建动画
        self.anim = animation.FuncAnimation(
            self.fig, 
            animate, 
            frames=len(self.times),
            interval=50,  # 20帧/秒
            blit=True
        )
        self.anim.running = True
        self.canvas.draw()
    
    def stop_animation(self):
        if hasattr(self, 'anim') and self.anim.running:
            self.anim.event_source.stop()
            # 移除标记线和标记点
            try:
                self.time_line1.remove()
                self.time_line2.remove()
                self.time_line3.remove()
                self.marker1.remove()
                self.marker2.remove()
                self.marker3.remove()
            except:
                pass
            self.anim.running = False
            self.canvas.draw()
    
    def save_results(self):
        if not self.filename or not self.formants:
            self.update_status("没有可保存的结果")
            return
        
        try:
            # 获取保存文件名
            base_name = os.path.splitext(os.path.basename(self.filename))[0]
            save_path = filedialog.asksaveasfilename(
                initialfile=f"{base_name}_formants",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if save_path:
                # 保存图像
                self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
                
                # 保存共振峰数据
                csv_formants_path = os.path.splitext(save_path)[0] + "_formants.csv"
                with open(csv_formants_path, 'w') as f:
                    f.write("Time(s),F1(Hz),F2(Hz),F3(Hz)\n")
                    for t, formants in zip(self.times, self.formants):
                        f1 = formants[0] if len(formants) > 0 else ""
                        f2 = formants[1] if len(formants) > 1 else ""
                        f3 = formants[2] if len(formants) > 2 else ""
                        f.write(f"{t:.4f},{f1},{f2},{f3}\n")
                
                # 保存语音波形数据
                csv_wave_path = os.path.splitext(save_path)[0] + "_waveform.csv"
                time_axis = np.arange(len(self.data)) / self.rate
                with open(csv_wave_path, 'w') as f:
                    f.write("Time(s),Amplitude\n")
                    for t, amp in zip(time_axis, self.data):
                        f.write(f"{t:.6f},{amp:.6f}\n")
                
                self.update_status(f"结果已保存:\n- 图像: {os.path.basename(save_path)}\n"
                                 f"- 共振峰数据: {os.path.basename(csv_formants_path)}\n"
                                 f"- 波形数据: {os.path.basename(csv_wave_path)}")
        except Exception as e:
            self.update_status(f"保存错误: {str(e)}")

# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicFormantAnalyzer(root)
    root.mainloop()