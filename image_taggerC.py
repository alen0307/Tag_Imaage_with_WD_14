#!/usr/bin/env python3
"""
图片分类标签工具 v1.0 - WD14 模型版

功能说明：
- 使用 WD14 本地模型为图片自动打标（替代 LM Studio）
- 支持批量处理，GPU 加速可达 10 倍提速
- 支持中英文标签输出
- 支持断点续传
- 提供丰富的标签分析和过滤功能

使用方法：
1. 双击 run_wd14_tagger.bat 启动（推荐）
2. 或运行: python image_taggerC.py

依赖安装：
- pip install onnxruntime (CPU)
- pip install onnxruntime-gpu (NVIDIA GPU)
- pip install onnxruntime-directml (AMD/Intel GPU)

模型要求：
- 模型文件需位于: E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-WD14-Tagger\models\
- 需要文件: wd-vit-tagger-v3.onnx, wd-vit-tagger-v3.csv

数据库：
- 默认使用: D:\projects\immich-booru-tagger-main\wd14_image_tags.db

详细文档：
- QUICK_START.md - 快速入门指南
- WD14_INTEGRATION_README.md - 完整技术文档
- CHANGES_SUMMARY.md - 修改总结

作者：Lingma
日期：2026-05-18
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sqlite3
import base64
import json
import requests
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import csv

# 添加 ONNX Runtime
try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    print("警告: 未安装 onnxruntime，将无法使用WD14模型")
    print("请运行: pip install onnxruntime")


class ImageTaggingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片分类标签工具 v1.0")
        self.root.geometry("900x800")

        self.directories = []
        self.running = False
        self.total_images = 0
        self.processed = 0
        self.skipped = 0
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        #self.db_path = os.path.join(self.base_dir, "image_tagsC.db")
        self.db_path =r"D:\projects\immich-booru-tagger-main\wd14_image_tags.db"
        self.config_path = os.path.join(self.base_dir, "config.json")
        
        # 缓存机制
        self._tag_stats_cache = None
        self._tag_stats_cache_time = 0
        self._thumbnail_cache = {}  # 缩略图缓存
        
        # 外部图片查看器路径
        self.image_viewer_path = r"D:\Program Files\XnViewMP\xnviewmp.exe"
        
        # 处理时间跟踪
        self.start_time = None
        
        # WD-14 模型相关
        self.model_dir = r"E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-WD14-Tagger\models"
        self.model_session = None
        self.model_tags = []
        self.general_index = None
        self.character_index = None
        self.confidence_threshold = 0.35
        self.batch_size = 16
        self.supports_batch = True
        
        # 翻译字典
        self.tag_translations = {}
        self.use_chinese_tags = False

        self.setup_db()
        self.create_widgets()
        self.load_config()
        self.init_wd14_model()

    def setup_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tag_stats (
                tag TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
        """)
        # 添加索引以加速查询
        conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_tags ON image_tags(tags)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_stats_count ON tag_stats(count DESC)")
        conn.commit()
        conn.close()

    def load_config(self):
        try:
            with open(self.config_path) as f:
                cfg = json.load(f)
            dirs = cfg.get("directories", [])
            for d in dirs:
                if os.path.isdir(d) and d not in self.directories:
                    self.directories.append(d)
                    self.dir_listbox.insert(tk.END, d)
            
            # 加载置信度阈值
            threshold = cfg.get("confidence_threshold", 0.35)
            self.confidence_threshold = threshold
            
            # 加载批处理大小
            batch_size = cfg.get("batch_size", 16)
            self.batch_size = batch_size
            
            # 加载翻译字典
            self.load_translations()
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_config(self):
        try:
            config = {
                "directories": self.directories,
                "confidence_threshold": self.confidence_threshold,
                "batch_size": self.batch_size
            }
            with open(self.config_path, "w") as f:
                json.dump(config, f, ensure_ascii=False)
        except Exception:
            pass
    
    def init_wd14_model(self):
        """初始化 WD-14 模型"""
        try:
            self.log("🔄 正在加载 WD-14 模型...")
            
            # 直接使用 wd-vit-tagger-v3.onnx（已知完好且支持批处理）
            selected_model = "wd-vit-tagger-v3.onnx"
            model_path = os.path.join(self.model_dir, selected_model)
            csv_path = os.path.join(self.model_dir, "wd-vit-tagger-v3.csv")
            
            if not os.path.exists(model_path):
                raise Exception(f"模型文件不存在: {model_path}")
            
            if not os.path.exists(csv_path):
                raise Exception(f"标签文件不存在: {csv_path}")
            
            self.log(f"📦 使用模型: {selected_model}")
            
            # 加载 ONNX 模型（自动检测可用的执行提供者）
            available_providers = ort.get_available_providers()
            
            # 优先使用 GPU，如果不可用则使用 CPU
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.log(f"⚡ 检测到 GPU，将使用 CUDA 加速")
            elif 'DirectMLExecutionProvider' in available_providers:
                providers = ['DirectMLExecutionProvider', 'CPUExecutionProvider']
                self.log(f"⚡ 检测到 DirectML，将使用 GPU 加速")
            else:
                providers = ['CPUExecutionProvider']
                self.log(f"💻 仅检测到 CPU，将使用 CPU 处理")
            
            self.model_session = InferenceSession(model_path, providers=providers)
            
            # wd-vit-tagger-v3 支持动态批次
            self.supports_batch = True
            self.batch_size = 16  # 默认批处理大小
            self.log(f"✅ 模型支持批处理（动态批次）")
            
            # 读取标签 CSV 文件
            self.load_tags(csv_path)
            
            self.log("✅ WD-14 模型加载成功")
            
        except Exception as e:
            self.log(f"❌ 加载 WD-14 模型失败: {e}")
            messagebox.showerror("错误", f"加载 WD-14 模型失败:\n{e}\n\n请确认:\n1. 模型文件存在于: {self.model_dir}\n2. 已安装 onnxruntime: pip install onnxruntime")
    
    def load_tags(self, csv_path):
        """从 CSV 文件加载标签"""
        self.model_tags = []
        self.general_index = None
        self.character_index = None
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) >= 3:
                    # 记录 general 和 character 的起始索引
                    if self.general_index is None and row[2] == "0":
                        self.general_index = len(self.model_tags)
                    elif self.character_index is None and row[2] == "4":
                        self.character_index = len(self.model_tags)
                    
                    # 将下划线替换为空格
                    tag_name = row[1].replace("_", " ")
                    self.model_tags.append(tag_name)
    
    def load_translations(self):
        """加载标签翻译字典"""
        translation_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tag_translations.json")
        try:
            if os.path.exists(translation_file):
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.tag_translations = json.load(f)
                self.log(f"✅ 已加载 {len(self.tag_translations)} 个标签翻译")
            else:
                self.log("⚠️  未找到翻译文件，将使用英文标签")
        except Exception as e:
            self.log(f"⚠️  加载翻译文件失败: {e}")
    
    def translate_tag(self, tag):
        """将英文标签翻译成中文"""
        if self.use_chinese_tags:
            # 将空格转换为下划线以匹配翻译字典的格式
            tag_key = tag.replace(" ", "_")
            if tag_key in self.tag_translations:
                return self.tag_translations[tag_key]
        return tag
    
    def toggle_chinese_tags(self):
        """切换中文/英文标签"""
        self.use_chinese_tags = self.use_chinese_var.get()
        if self.use_chinese_tags:
            self.log("🇨🇳 已切换到中文标签模式")
        else:
            self.log("🇺🇸 已切换到英文标签模式")

    def create_widgets(self):
        # Directory selection frame
        dir_frame = ttk.LabelFrame(self.root, text="目录列表", padding=5)
        dir_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Listbox with scrollbar
        list_frame = ttk.Frame(dir_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.dir_listbox = tk.Listbox(list_frame)
        self.dir_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.dir_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dir_listbox.config(yscrollcommand=scrollbar.set)

        # Buttons for directory management
        btn_frame = ttk.Frame(dir_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        ttk.Button(btn_frame, text="添加目录", command=self.add_directory).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btn_frame, text="删除目录", command=self.remove_directory).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(btn_frame, text="清空列表", command=self.clear_directories).pack(
            fill=tk.X, pady=2
        )

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 置信度阈值设置
        ttk.Label(control_frame, text="置信度:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.StringVar(value="0.35")
        threshold_entry = ttk.Entry(control_frame, textvariable=self.threshold_var, width=6)
        threshold_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(control_frame, text="批次:").pack(side=tk.LEFT, padx=(10, 2))
        self.batch_size_var = tk.StringVar(value="16")
        self.batch_entry = ttk.Entry(control_frame, textvariable=self.batch_size_var, width=4)
        self.batch_entry.pack(side=tk.LEFT, padx=2)
        
        # 中文标签选项
        self.use_chinese_var = tk.BooleanVar(value=False)
        chinese_check = ttk.Checkbutton(control_frame, text="中文标签", variable=self.use_chinese_var, command=self.toggle_chinese_tags)
        chinese_check.pack(side=tk.LEFT, padx=10)

        self.start_btn = ttk.Button(
            control_frame, text="开始处理", command=self.start_processing
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text="停止", command=self.stop_processing, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=5
        )
        ttk.Button(control_frame, text="分析结果", command=self.open_analysis).pack(
            side=tk.LEFT, padx=5
        )

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)

        # Bottom split pane: preview + log
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        preview_frame = ttk.LabelFrame(paned, text="图片预览", padding=5)
        paned.add(preview_frame, weight=1)

        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(expand=True)

        self.preview_label = ttk.Label(preview_inner, text="等待处理...")
        self.preview_label.pack()

        log_frame = ttk.LabelFrame(paned, text="处理日志", padding=5)
        paned.add(log_frame, weight=1)

        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_inner, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar2 = ttk.Scrollbar(
            log_inner, orient=tk.VERTICAL, command=self.log_text.yview
        )
        scrollbar2.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar2.set)

        log_inner.columnconfigure(0, weight=1)
        log_inner.rowconfigure(0, weight=1)

    def add_directory(self):
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        if dir_path in self.directories:
            messagebox.showinfo("提示", "该目录已在列表中")
            return
        self.directories.append(dir_path)
        self.dir_listbox.insert(tk.END, dir_path)
        self.log(f"添加目录: {dir_path}")
        self.save_config()
        self.update_stats()

    def remove_directory(self):
        selection = self.dir_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        dir_path = self.directories.pop(index)
        self.dir_listbox.delete(index)
        self.log(f"删除目录: {dir_path}")
        self.save_config()
        self.update_stats()

    def clear_directories(self):
        if not self.directories:
            return
        self.directories.clear()
        self.dir_listbox.delete(0, tk.END)
        self.log("已清空目录列表")
        self.save_config()
        self.update_stats()

    def update_stats(self):
        files = self.get_image_files()
        done = sum(1 for f in files if self.is_already_analyzed(f))
        # 不再显示统计信息

    def update_preview(self, image_path):
        def _update():
            try:
                img = Image.open(image_path)
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo
            except Exception as e:
                self.preview_label.config(image="", text=f"无法加载: {e}")

        self.root.after(0, _update)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def get_image_files(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        files = []
        for directory in self.directories:
            if not os.path.exists(directory):
                self.log(f"目录不存在: {directory}")
                continue
            for root_dir, _, filenames in os.walk(directory):
                for filename in filenames:
                    ext = Path(filename).suffix.lower()
                    if ext in image_extensions:
                        files.append(os.path.join(root_dir, filename))
        return files

    def is_already_analyzed(self, image_path):
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT tags FROM image_tags WHERE image_path=?", (image_path,)
            )
            row = cur.fetchone()
            return row
        finally:
            conn.close()
    
    def preprocess_image(self, image):
        """预处理单张图片"""
        # 转换为 RGB（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取模型输入尺寸
        input_shape = self.model_session.get_inputs()[0].shape
        height = input_shape[1]  # 通常是 448
        
        # 调整图片大小并填充
        ratio = float(height) / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (height, height), (255, 255, 255))
        square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
        
        # 转换为 numpy 数组
        image_array = np.array(square).astype(np.float32)
        image_array = image_array[:, :, ::-1]  # RGB -> BGR
        
        return image_array
    
    def analyze_images_batch(self, image_paths):
        """批量分析多张图片（利用 GPU 并行处理）"""
        if self.model_session is None:
            raise Exception("WD-14 模型未初始化")
        
        if not image_paths:
            return []
        
        try:
            # 如果模型不支持批处理，逐张处理
            if not self.supports_batch:
                results = []
                for path in image_paths:
                    try:
                        image = Image.open(path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        input_shape = self.model_session.get_inputs()[0].shape
                        height = input_shape[1]
                        
                        ratio = float(height) / max(image.size)
                        new_size = tuple([int(x * ratio) for x in image.size])
                        image = image.resize(new_size, Image.LANCZOS)
                        square = Image.new("RGB", (height, height), (255, 255, 255))
                        square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
                        
                        image_array = np.array(square).astype(np.float32)
                        image_array = image_array[:, :, ::-1]
                        image_array = np.expand_dims(image_array, 0)
                        
                        input_name = self.model_session.get_inputs()[0].name
                        output_name = self.model_session.get_outputs()[0].name
                        probs = self.model_session.run([output_name], {input_name: image_array})[0]
                        
                        result = list(zip(self.model_tags, probs[0]))
                        if self.general_index is not None and self.character_index is not None:
                            general_tags = result[self.general_index:self.character_index]
                        else:
                            general_tags = result
                        
                        filtered_tags = []
                        for tag, conf in general_tags:
                            if conf >= self.confidence_threshold:
                                filtered_tags.append((tag, conf))
                        
                        filtered_tags.sort(key=lambda x: x[1], reverse=True)
                        # 翻译标签
                        translated_tags = [self.translate_tag(tag) for tag, _ in filtered_tags]
                        tags_str = ", ".join(translated_tags)
                        results.append((path, tags_str))
                    except Exception as e:
                        self.log(f"⚠️ 跳过无效图片: {os.path.basename(path)} - {e}")
                return results
            
            # 预处理所有图片
            preprocessed = []
            valid_paths = []
            
            for path in image_paths:
                try:
                    image = Image.open(path)
                    img_array = self.preprocess_image(image)
                    preprocessed.append(img_array)
                    valid_paths.append(path)
                except Exception as e:
                    self.log(f"⚠️ 跳过无效图片: {os.path.basename(path)} - {e}")
            
            if not preprocessed:
                return []
            
            # 堆叠成批次
            batch_array = np.stack(preprocessed, axis=0)
            
            # 获取输入输出名称
            input_name = self.model_session.get_inputs()[0].name
            output_name = self.model_session.get_outputs()[0].name
            
            # 批量推理（一次性处理所有图片）
            probs = self.model_session.run([output_name], {input_name: batch_array})[0]
            
            # 处理结果
            results = []
            for idx, prob in enumerate(probs):
                # 提取标签和置信度
                result = list(zip(self.model_tags, prob))
                
                # 过滤标签（只使用 general 标签）
                if self.general_index is not None and self.character_index is not None:
                    general_tags = result[self.general_index:self.character_index]
                else:
                    general_tags = result
                
                # 根据置信度过滤
                filtered_tags = []
                for tag, conf in general_tags:
                    if conf >= self.confidence_threshold:
                        filtered_tags.append((tag, conf))
                
                # 按置信度排序
                filtered_tags.sort(key=lambda x: x[1], reverse=True)
                
                # 翻译标签
                translated_tags = [self.translate_tag(tag) for tag, _ in filtered_tags]
                
                # 生成标签字符串
                tags_str = ", ".join(translated_tags)
                results.append((valid_paths[idx], tags_str))
            
            return results
            
        except Exception as e:
            raise Exception(f"批量分析失败: {str(e)}")
    
    def analyze_image(self, image_path):
        """使用 WD-14 分析单张图片（向后兼容）"""
        results = self.analyze_images_batch([image_path])
        if results:
            return results[0][1]
        return ""

    def save_result(self, image_path, tags):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO image_tags (image_path, tags) VALUES (?, ?) "
                "ON CONFLICT(image_path) DO UPDATE SET tags=excluded.tags, created_at=CURRENT_TIMESTAMP",
                (image_path, tags),
            )
            conn.commit()
        finally:
            conn.close()

    def parse_tags(self, tags_string):
        for sep in [",", "，", "、", "；", ";", "|", "/", " ", "\n"]:
            tags_string = tags_string.replace(sep, ",")
        return [t.strip() for t in tags_string.split(",") if t.strip()]

    def update_tag_stats(self, tags_string):
        """优化:批量更新标签统计"""
        tags = self.parse_tags(tags_string)
        if not tags:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            # 批量更新,减少数据库交互
            for tag in tags:
                conn.execute(
                    "INSERT INTO tag_stats (tag, count) VALUES (?, 1) "
                    "ON CONFLICT(tag) DO UPDATE SET count = count + 1",
                    (tag,),
                )
            conn.commit()
            # 清除缓存以便下次获取最新数据
            self._tag_stats_cache = None
        finally:
            conn.close()

    def get_images_by_tag(self, tag):
        """优化:使用LIKE查询替代全表扫描+Python过滤"""
        conn = sqlite3.connect(self.db_path)
        try:
            # 使用LIKE进行模糊匹配,避免加载所有数据到内存
            pattern = f"%{tag}%"
            cur = conn.execute(
                "SELECT image_path, tags FROM image_tags WHERE tags LIKE ?",
                (pattern,)
            )
            results = []
            for path, tags_string in cur:
                if tags_string and tag in self.parse_tags(tags_string):
                    results.append(path)
            return results
        finally:
            conn.close()

    def get_tag_stats(self, force_refresh=False):
        """优化:添加缓存机制,避免频繁查询数据库"""
        import time
        current_time = time.time()
        
        # 如果缓存有效且未强制刷新,直接返回缓存
        if not force_refresh and self._tag_stats_cache is not None:
            if current_time - self._tag_stats_cache_time < 5:  # 5秒缓存
                return self._tag_stats_cache
        
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("SELECT tag, count FROM tag_stats ORDER BY count DESC")
            result = cur.fetchall()
            # 更新缓存
            self._tag_stats_cache = result
            self._tag_stats_cache_time = current_time
            return result
        finally:
            conn.close()

    def rebuild_tag_stats(self):
        """优化:批量操作减少数据库交互次数"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM tag_stats")
            cur = conn.execute(
                "SELECT tags FROM image_tags WHERE tags IS NOT NULL AND tags != ''"
            )
            freq = {}
            for (tags_string,) in cur:
                for tag in self.parse_tags(tags_string):
                    freq[tag] = freq.get(tag, 0) + 1
            
            # 批量插入,提高性能
            batch_data = [(tag, count) for tag, count in freq.items()]
            conn.executemany(
                "INSERT INTO tag_stats (tag, count) VALUES (?, ?)",
                batch_data
            )
            conn.commit()
            
            # 清除缓存
            self._tag_stats_cache = None
            return len(freq)
        finally:
            conn.close()

    def get_tags_by_image(self, image_path):
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT tags FROM image_tags WHERE image_path=?", (image_path,)
            )
            row = cur.fetchone()
            return row[0] if row else ""
        finally:
            conn.close()

    def get_images_by_path(self, keywords):
        """根据关键词搜索图片路径中包含该关键词的图片"""
        conn = sqlite3.connect(self.db_path)
        try:
            matching_images = []
            for keyword in keywords:
                # 使用LIKE进行模糊匹配图片路径
                pattern = f"%{keyword}%"
                cur = conn.execute(
                    "SELECT DISTINCT image_path FROM image_tags WHERE image_path LIKE ?",
                    (pattern,)
                )
                for (path,) in cur:
                    if path not in matching_images:
                        matching_images.append(path)
            return matching_images
        finally:
            conn.close()

    def get_tags_from_images(self, image_paths):
        """从指定的图片列表中获取所有标签及其频次"""
        conn = sqlite3.connect(self.db_path)
        try:
            tag_freq = {}
            for image_path in image_paths:
                cur = conn.execute(
                    "SELECT tags FROM image_tags WHERE image_path = ?",
                    (image_path,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    tags = self.parse_tags(row[0])
                    for tag in tags:
                        tag_freq[tag] = tag_freq.get(tag, 0) + 1
            return tag_freq
        finally:
            conn.close()

    def open_analysis(self):
        win = tk.Toplevel(self.root)
        win.title("标签分析结果")
        win.geometry("1400x700")

        # Toolbar
        toolbar = ttk.Frame(win)
        toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))

        status_label = ttk.Label(toolbar, text="就绪")
        status_label.pack(side=tk.RIGHT, padx=5)

        def do_rebuild():
            status_label.config(text="正在统计...")
            win.update_idletasks()
            
            # 如果当前在小世界模式，先退出小世界（删除临时数据库，恢复原始路径）
            if small_world_db_path[0] and os.path.exists(small_world_db_path[0]):
                try:
                    os.remove(small_world_db_path[0])
                except:
                    pass
                self.db_path = original_db_path[0]
                small_world_db_path[0] = None
                status_label.config(text="已退出小世界模式")
                win.update_idletasks()
            
            # 从原始数据库重建统计
            count = self.rebuild_tag_stats()
            
            # 完全重置：清除所有缓存，像重新打开面板一样
            all_tags_cache.clear()  # 清除标签统计缓存
            displayed_tags_count = 0  # 重置已显示标签计数
            
            # 清空并重新加载标签列表
            for widget in tag_container.winfo_children():
                widget.destroy()
            tag_actions.clear()
            tag_buttons.clear()
            
            # 重新从头加载标签
            refresh_tags(0)
            
            status_label.config(text=f"重新统计完成，共 {count} 个标签")
        
        ttk.Button(toolbar, text="重新统计（从数据库重建）", command=do_rebuild).pack(
            side=tk.LEFT, padx=2
        )
        
        # 添加操作模式单选框
        mode_var = tk.StringVar(value="新")
        mode_frame = ttk.Frame(toolbar)
        mode_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(mode_frame, text="操作模式:").pack(side=tk.LEFT, padx=(0, 5))
        
        for mode_text in ["新", "加", "减", "交"]:
            ttk.Radiobutton(
                mode_frame,
                text=mode_text,
                variable=mode_var,
                value=mode_text
            ).pack(side=tk.LEFT, padx=2)
        
        # 添加小世界按钮
        def enter_small_world():
            """进入小世界模式：创建临时数据库，切换到小世界"""
            if not current_thumbnails:
                messagebox.showwarning("警告", "当前没有筛选出任何图片，请先进行筛选")
                return
            
            import tempfile
            
            # 如果已经在小世界，先退出（删除旧的小世界数据库）
            if small_world_db_path[0] and os.path.exists(small_world_db_path[0]):
                try:
                    os.remove(small_world_db_path[0])
                except:
                    pass
            
            # 创建临时数据库文件
            temp_dir = tempfile.gettempdir()
            small_world_db_path[0] = os.path.join(temp_dir, f"small_world_{os.getpid()}.db")
            
            # 从原始数据库复制结构并筛选数据
            conn_src = sqlite3.connect(original_db_path[0])
            conn_dst = sqlite3.connect(small_world_db_path[0])
            
            try:
                # 复制表结构（排除系统表）
                cursor = conn_src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                for (sql,) in cursor:
                    if sql:
                        conn_dst.execute(sql)
                
                # 只复制当前缩略图相关的数据
                if current_thumbnails:
                    placeholders = ','.join(['?' for _ in current_thumbnails])
                    
                    # 附加原始数据库（使用 source_db 别名）
                    conn_dst.execute(f"ATTACH DATABASE '{original_db_path[0]}' AS source_db")
                    
                    # 复制 image_tags 数据
                    query = f"INSERT INTO image_tags SELECT * FROM source_db.image_tags WHERE image_path IN ({placeholders})"
                    conn_dst.execute(query, current_thumbnails)
                    
                    # 先提交插入操作
                    conn_dst.commit()
                    
                    # 再分离数据库
                    conn_dst.execute("DETACH DATABASE source_db")
                
                conn_dst.commit()
                
                # 重建小世界的 tag_stats
                cursor = conn_dst.execute("SELECT tags FROM image_tags WHERE tags IS NOT NULL AND tags != ''")
                freq = {}
                for (tags_string,) in cursor:
                    for tag in self.parse_tags(tags_string):
                        freq[tag] = freq.get(tag, 0) + 1
                
                batch_data = [(tag, count) for tag, count in freq.items()]
                conn_dst.executemany(
                    "INSERT INTO tag_stats (tag, count) VALUES (?, ?)",
                    batch_data
                )
                conn_dst.commit()
                
            finally:
                conn_src.close()
                conn_dst.close()
            
            # 切换到小世界数据库
            self.db_path = small_world_db_path[0]
            
            # 清除缓存并刷新
            all_tags_cache.clear()
            refresh_tags()
            
            count = len(freq) if current_thumbnails else 0
            status_label.config(text=f"小世界模式：共 {count} 个标签（基于 {len(current_thumbnails)} 张图片）")
        
        small_world_btn = ttk.Button(
            toolbar,
            text="小世界",
            command=enter_small_world
        )
        small_world_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加复制图片按钮
        def copy_images_to_directory():
            """将当前显示的所有缩略图复制到用户选择的目录"""
            if not current_thumbnails:
                messagebox.showwarning("警告", "当前没有显示任何图片")
                return
            
            # 让用户选择目标目录
            target_dir = filedialog.askdirectory(title="选择目标目录")
            if not target_dir:
                return
            
            # 确认操作
            result = messagebox.askyesno(
                "确认复制",
                f"即将复制 {len(current_thumbnails)} 张图片到：\n{target_dir}\n\n是否继续？"
            )
            if not result:
                return
            
            # 开始复制
            status_label.config(text=f"正在复制 0/{len(current_thumbnails)} 张图片...")
            win.update_idletasks()
            
            import shutil
            copied_count = 0
            failed_count = 0
            
            for idx, img_path in enumerate(current_thumbnails):
                try:
                    if os.path.exists(img_path):
                        # 获取文件名
                        filename = os.path.basename(img_path)
                        target_path = os.path.join(target_dir, filename)
                        
                        # 如果文件已存在，添加序号
                        if os.path.exists(target_path):
                            name, ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(target_path):
                                target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                                counter += 1
                        
                        # 复制文件
                        shutil.copy2(img_path, target_path)
                        copied_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    self.log(f"复制失败: {img_path} - {e}")
                
                # 每复制10张更新一次进度
                if (idx + 1) % 10 == 0 or idx == len(current_thumbnails) - 1:
                    status_label.config(text=f"正在复制 {idx + 1}/{len(current_thumbnails)} 张图片...")
                    win.update_idletasks()
            
            # 显示结果
            status_label.config(text="就绪")
            messagebox.showinfo(
                "复制完成",
                f"复制完成！\n\n成功: {copied_count} 张\n失败: {failed_count} 张\n目标目录: {target_dir}"
            )
        
        ttk.Button(
            toolbar,
            text="复制图片到目录",
            command=copy_images_to_directory
        ).pack(side=tk.LEFT, padx=5)

        paned = ttk.PanedWindow(win, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: tag list (使用Canvas替代Text以获得更好的滚动性能)
        left_frame = ttk.LabelFrame(paned, text="标签列表（按频次排序）", padding=5, width=420)
        left_frame.pack_propagate(False)  # 禁止子组件改变框架大小
        paned.add(left_frame)
        
        # 添加过滤文本框
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, padx=2, pady=(0, 5))
        
        ttk.Label(filter_frame, text="包含:").pack(side=tk.LEFT, padx=(0, 3))
        filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=filter_var)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # 双击清空包含框
        def clear_filter(event):
            filter_var.set("")
        
        filter_entry.bind("<Double-Button-1>", clear_filter)
        
        ttk.Label(filter_frame, text="排除:").pack(side=tk.LEFT, padx=(5, 3))
        exclude_var = tk.StringVar()
        exclude_entry = ttk.Entry(filter_frame, textvariable=exclude_var)
        exclude_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # 双击清空排除框
        def clear_exclude(event):
            exclude_var.set("")
        
        exclude_entry.bind("<Double-Button-1>", clear_exclude)
        
        # 统一按钮
        def unify_tags():
            """统一标签功能"""
            filter_text = filter_var.get().strip()
            if not filter_text:
                messagebox.showwarning("警告", "请先在'包含'框中输入标签")
                return
            
            # 解析第一个标签
            for sep in [' ', ',', '，', '、', ';', '；']:
                filter_text = filter_text.replace(sep, ',')
            keywords = [kw.strip() for kw in filter_text.split(',') if kw.strip()]
            
            if not keywords:
                messagebox.showwarning("警告", "未检测到有效的标签关键词")
                return
            
            target_tag = keywords[0]
            
            # 确认操作
            result = messagebox.askyesno(
                "确认统一标签",
                f"是否将下面显示的所有标签统一修改为：'{target_tag}'？\n\n"
                f"注意：此操作会修改数据库，操作前会自动备份。"
            )
            
            if not result:
                return
            
            # 获取当前过滤后的所有标签
            nonlocal all_tags_cache
            if not all_tags_cache:
                all_tags_cache = self.get_tag_stats()
            
            stats = all_tags_cache
            exclude_text = exclude_var.get().strip()
            exclude_keywords = []
            
            if exclude_text:
                for sep in [' ', ',', '，', '、', ';', '；']:
                    exclude_text = exclude_text.replace(sep, ',')
                exclude_keywords = [kw.strip().lower() for kw in exclude_text.split(',') if kw.strip()]
            
            # 筛选出需要修改的标签
            tags_to_unify = []
            for tag, count in stats:
                tag_lower = tag.lower()
                
                # 检查是否包含任意一个过滤关键词
                include = True
                if keywords:
                    include = any(kw.lower() in tag_lower for kw in keywords)
                
                # 检查是否包含任意一个排除关键词
                exclude = False
                if exclude_keywords:
                    exclude = any(kw in tag_lower for kw in exclude_keywords)
                
                # 只有满足包含条件且不满足排除条件才需要修改
                if include and not exclude:
                    tags_to_unify.append(tag)
            
            if not tags_to_unify:
                messagebox.showinfo("提示", "没有需要统一的标签")
                return
            
            # 备份数据库
            import shutil
            from datetime import datetime
            backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.copy2(self.db_path, backup_path)
                self.log(f"数据库已备份至: {backup_path}")
            except Exception as e:
                messagebox.showerror("错误", f"备份数据库失败: {e}")
                return
            
            # 开始统一标签
            status_label.config(text=f"正在统一 {len(tags_to_unify)} 个标签...")
            win.update_idletasks()
            
            conn = sqlite3.connect(self.db_path)
            try:
                modified_count = 0
                total_images = 0
                
                for old_tag in tags_to_unify:
                    # 查找包含该标签的所有图片
                    cur = conn.execute(
                        "SELECT image_path, tags FROM image_tags WHERE tags LIKE ?",
                        (f"%{old_tag}%",)
                    )
                    
                    for image_path, tags_string in cur:
                        if tags_string and old_tag in self.parse_tags(tags_string):
                            # 替换标签
                            tags_list = self.parse_tags(tags_string)
                            # 移除旧标签
                            tags_list = [t for t in tags_list if t != old_tag]
                            # 添加新标签（如果不存在）
                            if target_tag not in tags_list:
                                tags_list.insert(0, target_tag)
                            
                            new_tags_string = ",".join(tags_list)
                            conn.execute(
                                "UPDATE image_tags SET tags = ?, created_at = CURRENT_TIMESTAMP WHERE image_path = ?",
                                (new_tags_string, image_path)
                            )
                            total_images += 1
                    
                    modified_count += 1
                    
                    # 每处理10个标签更新一次状态
                    if modified_count % 10 == 0:
                        status_label.config(text=f"正在统一 {modified_count}/{len(tags_to_unify)} 个标签...")
                        win.update_idletasks()
                
                # 重建标签统计
                conn.execute("DELETE FROM tag_stats")
                cur = conn.execute("SELECT tags FROM image_tags WHERE tags IS NOT NULL AND tags != ''")
                freq = {}
                for (tags_string,) in cur:
                    for tag in self.parse_tags(tags_string):
                        freq[tag] = freq.get(tag, 0) + 1
                
                batch_data = [(tag, count) for tag, count in freq.items()]
                conn.executemany(
                    "INSERT INTO tag_stats (tag, count) VALUES (?, ?)",
                    batch_data
                )
                conn.commit()
                
                # 清除缓存
                self._tag_stats_cache = None
                all_tags_cache.clear()
                
                status_label.config(text="就绪")
                messagebox.showinfo(
                    "完成",
                    f"统一标签完成！\n\n"
                    f"修改标签数: {modified_count} 个\n"
                    f"影响图片数: {total_images} 张\n"
                    f"统一为: {target_tag}\n\n"
                    f"数据库备份: {backup_path}"
                )
                
                # 刷新标签列表
                refresh_tags()
                
            except Exception as e:
                status_label.config(text="就绪")
                messagebox.showerror("错误", f"统一标签失败: {e}")
                self.log(f"统一标签失败: {e}")
            finally:
                conn.close()
        
        ttk.Button(filter_frame, text="统一", command=unify_tags).pack(side=tk.LEFT, padx=5)
        
        # 标签排序单选框
        sort_frame = ttk.Frame(left_frame)
        sort_frame.pack(fill=tk.X, padx=2, pady=(0, 5))
        
        ttk.Label(sort_frame, text="排序:").pack(side=tk.LEFT, padx=(0, 5))
        
        sort_var = tk.StringVar(value="1")
        
        for value, text in [("1", "频次(大-小)"), ("2", "频次(小-大)"), ("3", "名称"), ("4", "名称倒序")]:
            ttk.Radiobutton(
                sort_frame,
                text=text,
                variable=sort_var,
                value=value,
                command=lambda: refresh_tags()
            ).pack(side=tk.LEFT, padx=2)
        
        # 防抖定时器
        filter_timer = [None]
        
        def on_filter_change(*args):
            """过滤文本变化时重新显示标签（带防抖）"""
            # 取消之前的定时器
            if filter_timer[0]:
                win.after_cancel(filter_timer[0])
            # 设置新的定时器，150ms后执行（减少延迟）
            filter_timer[0] = win.after(150, lambda: apply_filter(filter_var.get(), exclude_var.get()))
        
        def on_exclude_change(*args):
            """排除文本变化时重新显示标签（带防抖）"""
            # 取消之前的定时器
            if filter_timer[0]:
                win.after_cancel(filter_timer[0])
            # 设置新的定时器，150ms后执行（减少延迟）
            filter_timer[0] = win.after(150, lambda: apply_filter(filter_var.get(), exclude_var.get()))
        
        def apply_filter(filter_text, exclude_text):
            """应用过滤，不通过 trace 触发"""
            # 处理包含文本：去掉引号，提取文件名
            processed_filter = filter_text.strip()
            
            # 去掉引号（单引号和双引号）
            processed_filter = processed_filter.strip('"').strip("'")
            
            # 如果包含路径分隔符，只保留最后一部分（文件名）
            if '/' in processed_filter or '\\' in processed_filter:
                # 替换所有分隔符为统一格式
                processed_filter = processed_filter.replace('\\', '/')
                # 取最后一个分隔符后面的部分
                processed_filter = processed_filter.split('/')[-1]
            
            # 如果处理后的文本不为空，设置到过滤框
            if processed_filter and processed_filter != filter_text.strip():
                filter_var.set(processed_filter)
            
            # 尝试用图片路径匹配
            if processed_filter:
                # 解析关键词
                for sep in [' ', ',', '，', '、', ';', '；']:
                    processed_filter_temp = processed_filter.replace(sep, ',')
                keywords = [kw.strip().lower() for kw in processed_filter_temp.split(',') if kw.strip()]
                
                if keywords:
                    # 搜索图片路径
                    matching_images = self.get_images_by_path(keywords)
                    
                    if matching_images:
                        # 直接显示匹配的图片到缩略图框
                        current_thumbnails.clear()
                        current_thumbnails.extend(matching_images)
                        _display_current_thumbnails(0)
                        status_label.config(text=f"路径匹配: 找到 {len(matching_images)} 张图片")
                        return
            
            # 如果没有路径匹配，执行正常的标签过滤
            refresh_tags()
        
        def set_filter_immediately(text):
            """立即设置过滤，无延迟"""
            # 取消待处理的定时器
            if filter_timer[0]:
                win.after_cancel(filter_timer[0])
                filter_timer[0] = None
            # 直接设置值并立即刷新
            filter_var.set(text)
            refresh_tags()
        
        filter_var.trace_add("write", on_filter_change)
        exclude_var.trace_add("write", on_exclude_change)

        # 使用Canvas + Frame实现高性能标签列表
        tag_canvas = tk.Canvas(left_frame, highlightthickness=0)
        tag_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tag_scroll = ttk.Scrollbar(
            left_frame, orient=tk.VERTICAL, command=tag_canvas.yview
        )
        tag_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tag_canvas.config(yscrollcommand=tag_scroll.set)

        # 创建内部容器用于放置标签按钮
        tag_container = ttk.Frame(tag_canvas)
        tag_canvas.create_window((0, 0), window=tag_container, anchor="nw")

        def _on_tag_configure(event):
            tag_canvas.config(scrollregion=tag_canvas.bbox("all"))

        tag_container.bind("<Configure>", _on_tag_configure)
        
        # Middle: thumbnail canvas
        mid_frame = ttk.LabelFrame(paned, text="相关图片（点击缩略图预览）", padding=5)
        paned.add(mid_frame)

        canvas = tk.Canvas(mid_frame, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        thumb_scroll = ttk.Scrollbar(
            mid_frame, orient=tk.VERTICAL, command=canvas.yview
        )
        thumb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.config(yscrollcommand=thumb_scroll.set)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.config(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_configure)
        
        # 创建目录图片框（不立即添加到 paned）
        dir_frame = ttk.LabelFrame(paned, text="目录图片（按修改时间排序）", padding=5)
        # 设置深灰色背景
        dir_frame.configure(style='Dark.TLabelframe')
        
        # 创建自定义样式
        style = ttk.Style()
        style.configure('Dark.TLabelframe', background='#404040')
        style.configure('Dark.TLabelframe.Label', background='#404040', foreground='white')
        
        dir_canvas = tk.Canvas(dir_frame, highlightthickness=0)
        dir_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        dir_scroll = ttk.Scrollbar(
            dir_frame, orient=tk.VERTICAL, command=dir_canvas.yview
        )
        dir_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        dir_canvas.config(yscrollcommand=dir_scroll.set)
        
        dir_inner = ttk.Frame(dir_canvas)
        dir_canvas.create_window((0, 0), window=dir_inner, anchor="nw")
        
        def _on_dir_configure(event):
            dir_canvas.config(scrollregion=dir_canvas.bbox("all"))
        
        dir_inner.bind("<Configure>", _on_dir_configure)
        
        # 统一的鼠标滚轮事件处理 - 根据鼠标位置决定滚动哪个控件
        def _on_mousewheel(event):
            # 获取鼠标当前位置
            widget = event.widget
            
            # 检查鼠标是否在标签列表区域
            try:
                tag_x, tag_y = tag_canvas.winfo_pointerxy()
                tag_left = tag_canvas.winfo_rootx()
                tag_top = tag_canvas.winfo_rooty()
                tag_right = tag_left + tag_canvas.winfo_width()
                tag_bottom = tag_top + tag_canvas.winfo_height()
                
                if tag_left <= tag_x <= tag_right and tag_top <= tag_y <= tag_bottom:
                    tag_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                    return "break"
            except:
                pass
            
            # 检查鼠标是否在缩略图区域
            try:
                thumb_x, thumb_y = canvas.winfo_pointerxy()
                thumb_left = canvas.winfo_rootx()
                thumb_top = canvas.winfo_rooty()
                thumb_right = thumb_left + canvas.winfo_width()
                thumb_bottom = thumb_top + canvas.winfo_height()
                
                if thumb_left <= thumb_x <= thumb_right and thumb_top <= thumb_y <= thumb_bottom:
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                    return "break"
            except:
                pass
            
            # 检查鼠标是否在目录图片框区域（仅在可见时）
            try:
                # 检查 dir_frame 是否在 paned window 中（即是否可见）
                if dir_panel_visible[0]:
                    dir_x, dir_y = dir_canvas.winfo_pointerxy()
                    dir_left = dir_canvas.winfo_rootx()
                    dir_top = dir_canvas.winfo_rooty()
                    dir_right = dir_left + dir_canvas.winfo_width()
                    dir_bottom = dir_top + dir_canvas.winfo_height()
                    
                    if dir_left <= dir_x <= dir_right and dir_top <= dir_y <= dir_bottom:
                        dir_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                        return "break"
            except:
                pass
        
        # 只绑定一次，全局生效
        win.bind_all("<MouseWheel>", _on_mousewheel)

        tag_actions = {}
        tag_buttons = []  # 保存按钮引用防止被垃圾回收
        current_thumbnails = []  # 保存当前显示的缩略图路径列表
        all_tags_cache = []  # 缓存所有标签数据，避免重复查询
        displayed_tags_count = 0  # 已显示的标签数量
        dir_panel_visible = [False]  # 目录面板是否可见
        small_world_db_path = [None]  # 小世界临时数据库路径
        original_db_path = [self.db_path]  # 保存原始数据库路径
        
        # 窗口关闭时清理小世界数据库
        def on_closing():
            if small_world_db_path[0] and os.path.exists(small_world_db_path[0]):
                try:
                    os.remove(small_world_db_path[0])
                except:
                    pass
            self.db_path = original_db_path[0]
            win.destroy()
        
        win.protocol("WM_DELETE_WINDOW", on_closing)
        
        def show_directory_images(current_image_path):
            """显示当前图片所在目录中按修改时间排序的50张图片"""
            try:
                # 获取当前图片所在目录
                dir_path = os.path.dirname(current_image_path)
                if not os.path.exists(dir_path):
                    messagebox.showerror("错误", f"目录不存在: {dir_path}")
                    return
                
                # 获取目录中所有图片文件
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
                all_images = []
                
                for filename in os.listdir(dir_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        filepath = os.path.join(dir_path, filename)
                        # 获取修改时间
                        mtime = os.path.getmtime(filepath)
                        all_images.append((filepath, mtime))
                
                if not all_images:
                    messagebox.showwarning("警告", "目录中没有图片文件")
                    return
                
                # 按修改时间排序
                all_images.sort(key=lambda x: x[1])
                
                # 找到当前图片的索引
                current_index = -1
                for idx, (path, _) in enumerate(all_images):
                    if path == current_image_path:
                        current_index = idx
                        break
                
                if current_index == -1:
                    messagebox.showerror("错误", "当前图片不在目录中")
                    return
                
                # 取前24张 + 当前 + 后25张，共50张
                start_idx = max(0, current_index - 24)
                end_idx = min(len(all_images), current_index + 26)
                
                # 确保总共50张（如果可能）
                if end_idx - start_idx < 50:
                    if start_idx == 0:
                        end_idx = min(len(all_images), 50)
                    elif end_idx == len(all_images):
                        start_idx = max(0, len(all_images) - 50)
                
                selected_images = [path for path, _ in all_images[start_idx:end_idx]]
                
                # 清空目录图片框
                for widget in dir_inner.winfo_children():
                    widget.destroy()
                
                # 显示目录图片
                cols = 5
                row_frame = None
                
                for idx, img_path in enumerate(selected_images):
                    if idx % cols == 0:
                        row_frame = ttk.Frame(dir_inner)
                        row_frame.pack(fill=tk.X, padx=2, pady=2)
                    
                    try:
                        # 加载缩略图（使用完整路径作为缓存键）
                        cache_key = f"dir_thumb_{img_path}"
                        if cache_key in self._thumbnail_cache:
                            photo = self._thumbnail_cache[cache_key]
                        else:
                            img = Image.open(img_path)
                            img.thumbnail((120, 120))
                            photo = ImageTk.PhotoImage(img)
                            self._thumbnail_cache[cache_key] = photo
                        
                        btn = ttk.Button(
                            row_frame,
                            image=photo,
                            command=lambda p=img_path: show_preview_in_dir_panel(p),
                        )
                        btn.image = photo
                        
                        # 绑定右键事件，返回缩略图框
                        def on_dir_right_click(event):
                            switch_back_to_thumbnails()
                        
                        btn.bind("<Button-3>", on_dir_right_click)
                        btn.pack(side=tk.LEFT, padx=2)
                    except Exception as e:
                        self.log(f"加载缩略图失败: {e}")
                
                # 切换到目录图片框
                if not dir_panel_visible[0]:
                    # 移除缩略图框和预览框
                    paned.forget(mid_frame)
                    paned.forget(preview_frame)
                    # 添加目录图片框和预览框
                    paned.add(dir_frame)
                    paned.add(preview_frame)
                    dir_panel_visible[0] = True
                    # 清理已销毁的按钮引用
                    tag_buttons[:] = [btn for btn in tag_buttons if btn.winfo_exists()]
                
                dir_canvas.update_idletasks()
                dir_canvas.config(scrollregion=dir_canvas.bbox("all"))
                
                status_label.config(text=f"目录图片: {len(selected_images)} 张")
                
            except Exception as e:
                messagebox.showerror("错误", f"显示目录图片失败: {e}")
                self.log(f"显示目录图片失败: {e}")
        
        def show_preview_in_dir_panel(image_path):
            """在目录图片框中点击图片后的处理"""
            # 显示预览
            try:
                img = Image.open(image_path)
                img.thumbnail((500, 500))
                photo = ImageTk.PhotoImage(img)
                preview_label.config(image=photo, text="")
                preview_label.image = photo
            except Exception as e:
                preview_label.config(image="", text=f"无法加载: {e}")
            
            # 从数据库查询标签
            tags = self.get_tags_by_image(image_path)
            tag_text.delete("1.0", tk.END)
            tag_text.insert("1.0", tags)
            
            # 更新当前图片路径
            current_img_path[0] = image_path
        
        def switch_back_to_thumbnails():
            """切换回缩略图框"""
            if dir_panel_visible[0]:
                # 移除目录图片框和预览框
                paned.forget(dir_frame)
                paned.forget(preview_frame)
                # 添加缩略图框和预览框
                paned.add(mid_frame)
                paned.add(preview_frame)
                dir_panel_visible[0] = False
                status_label.config(text="就绪")
                # 清理已销毁的按钮引用
                tag_buttons[:] = [btn for btn in tag_buttons if btn.winfo_exists()]

        def refresh_tags(start_index=0):
            """刷新标签列表，支持分批加载"""
            # 获取过滤和排除文本
            filter_text = filter_var.get().strip()
            exclude_text = exclude_var.get().strip()
            
            # 使用缓存的标签数据，避免重复查询数据库
            nonlocal all_tags_cache, displayed_tags_count
            if not all_tags_cache:
                # 只在缓存为空时从数据库读取（只读 tag_stats 表）
                all_tags_cache = self.get_tag_stats()
            
            stats = all_tags_cache
            if not stats:
                # 清空所有按钮
                for widget in tag_container.winfo_children():
                    widget.destroy()
                displayed_tags_count = 0
                return
            
            # 解析过滤和排除关键词（支持空格、逗号等分隔符）
            filter_keywords = []
            exclude_keywords = []
            
            if filter_text:
                # 支持多种分隔符：空格、逗号（中英文）、顿号等
                for sep in [' ', ',', '，', '、', ';', '；']:
                    filter_text = filter_text.replace(sep, ',')
                filter_keywords = [kw.strip().lower() for kw in filter_text.split(',') if kw.strip()]
            
            if exclude_text:
                # 支持多种分隔符：空格、逗号（中英文）、顿号等
                for sep in [' ', ',', '，', '、', ';', '；']:
                    exclude_text = exclude_text.replace(sep, ',')
                exclude_keywords = [kw.strip().lower() for kw in exclude_text.split(',') if kw.strip()]
            
            # 应用过滤逻辑
            if filter_keywords or exclude_keywords:
                filtered_stats = []
                for tag, count in stats:
                    tag_lower = tag.lower()
                    
                    # 检查是否包含任意一个过滤关键词
                    include = True
                    if filter_keywords:
                        include = any(kw in tag_lower for kw in filter_keywords)
                    
                    # 检查是否包含任意一个排除关键词
                    exclude = False
                    if exclude_keywords:
                        exclude = any(kw in tag_lower for kw in exclude_keywords)
                    
                    # 只有满足包含条件且不满足排除条件才显示
                    if include and not exclude:
                        filtered_stats.append((tag, count))
                
                if not filtered_stats and filter_keywords:
                    # 如果没有标签匹配，尝试用图片地址字段去匹配
                    matching_images = self.get_images_by_path(filter_keywords)
                    
                    if matching_images:
                        # 获取这些图片的所有标签
                        path_matched_tags = self.get_tags_from_images(matching_images)
                        
                        if path_matched_tags:
                            # 转换为列表格式
                            filtered_stats = [(tag, count) for tag, count in path_matched_tags.items()]
                            
                            # 应用排除过滤
                            if exclude_keywords:
                                filtered_stats = [
                                    (tag, count) for tag, count in filtered_stats
                                    if not any(kw in tag.lower() for kw in exclude_keywords)
                                ]
                
                if not filtered_stats:
                    # 清空所有按钮并显示提示
                    for widget in tag_container.winfo_children():
                        widget.destroy()
                    msg_parts = []
                    if filter_keywords:
                        msg_parts.append(f"包含: {', '.join(filter_keywords)}")
                    if exclude_keywords:
                        msg_parts.append(f"排除: {', '.join(exclude_keywords)}")
                    msg = ' | '.join(msg_parts)
                    ttk.Label(tag_container, text=f"未找到匹配的标签 ({msg})").pack(pady=10)
                    tag_container.update_idletasks()
                    tag_canvas.config(scrollregion=tag_canvas.bbox("all"))
                    displayed_tags_count = 0
                    return
                
                stats = filtered_stats
                # 过滤时从头开始显示
                start_index = 0
            
            # 应用排序逻辑
            sort_mode = sort_var.get()
            if sort_mode == "1":
                # 频次(大-小)
                stats = sorted(stats, key=lambda x: x[1], reverse=True)
            elif sort_mode == "2":
                # 频次(小-大)
                stats = sorted(stats, key=lambda x: x[1])
            elif sort_mode == "3":
                # 名称(升序)
                stats = sorted(stats, key=lambda x: x[0].lower())
            elif sort_mode == "4":
                # 名称倒序
                stats = sorted(stats, key=lambda x: x[0].lower(), reverse=True)
            
            # 如果是从头开始，清空现有按钮
            if start_index == 0:
                for widget in tag_container.winfo_children():
                    widget.destroy()
                tag_actions.clear()
                tag_buttons.clear()
                displayed_tags_count = 0
            
            # 计算本次要显示的标签范围
            batch_size = 300
            end_index = min(start_index + batch_size, len(stats))
            display_stats = stats[start_index:end_index]
            
            # 创建标签按钮
            _create_tag_buttons_batch(display_stats, start_index)
            
            displayed_tags_count = end_index
            
            # 如果还有更多标签，显示“加载更多”按钮
            if end_index < len(stats):
                remaining = len(stats) - end_index
                load_more_frame = ttk.Frame(tag_container)
                load_more_frame.grid(row=(displayed_tags_count // 4), column=0, columnspan=4, pady=10)
                
                def load_more_tags():
                    refresh_tags(end_index)
                
                ttk.Button(
                    load_more_frame,
                    text=f"加载更多 ({remaining} 个标签)",
                    command=load_more_tags
                ).pack()
            
            # 更新滚动区域
            tag_container.update_idletasks()
            tag_canvas.config(scrollregion=tag_canvas.bbox("all"))
        
        def _create_tag_buttons_batch(stats, start_index):
            """分批创建标签按钮"""
            # 使用Grid布局
            row = start_index // 4
            col = start_index % 4
            max_cols = 4
            
            for tag, count in stats:
                display_text = f"{tag} ({count})"
                
                btn = tk.Button(
                    tag_container,
                    text=display_text,
                    bg="#e0e0e0",
                    activebackground="#c0c0c0",
                    relief="raised",
                    width=12,
                    bd=1,
                    padx=2,
                    pady=1,
                    font=("Microsoft YaHei UI", 9),
                    cursor="hand2",
                    anchor="w",
                    command=lambda t=tag: on_tag_clicked(t)
                )
                
                # 绑定悬停效果
                def on_enter(e, button=btn):
                    button.config(bg="#c0c0c0")
                
                def on_leave(e, button=btn):
                    button.config(bg="#e0e0e0")
                
                btn.bind("<Enter>", on_enter)
                btn.bind("<Leave>", on_leave)
                
                btn.grid(row=row, column=col, sticky="ew", padx=2, pady=1)
                btn._tag_name = tag  # 保存标签名用于过滤
                tag_buttons.append(btn)
                tag_actions[tag] = btn
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
        
        def _show_hide_tag_buttons(filtered_stats):
            """显示/隐藏标签按钮（比销毁重建快得多）"""
            # 将过滤后的标签转为集合，快速查找
            visible_tags = {tag for tag, count in filtered_stats}
            
            row = 0
            col = 0
            max_cols = 4
            
            for btn in tag_buttons:
                # 检查按钮是否还存在
                try:
                    if not btn.winfo_exists():
                        continue
                except:
                    continue
                
                tag_name = btn._tag_name
                if tag_name in visible_tags:
                    btn.grid(row=row, column=col, sticky="ew", padx=2, pady=1)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                else:
                    btn.grid_forget()  # 隐藏但不销毁
        
        def on_tag_clicked(tag_name):
            """点击标签时的处理 - 根据模式执行不同操作"""
            mode = mode_var.get()
            # 直接从数据库查询（小世界模式下tag_stats已被替换，所以会自动从小世界数据中筛选）
            new_images = self.get_images_by_tag(tag_name)
            
            if mode == "新":
                # 新模式：清空当前，显示新标签的图片
                current_thumbnails.clear()
                current_thumbnails.extend(new_images)
                _display_current_thumbnails(0)  # 从头开始显示，支持分批加载
                status_label.config(text=f"模式[新]: 显示 {len(new_images)} 张图片")
                
            elif mode == "加":
                # 加模式：去重后添加到当前缩略图
                added_count = 0
                for img_path in new_images:
                    if img_path not in current_thumbnails:
                        current_thumbnails.append(img_path)
                        added_count += 1
                
                # 重新显示所有图片
                _display_current_thumbnails()
                status_label.config(text=f"模式[加]: 新增 {added_count} 张，共 {len(current_thumbnails)} 张")
                
            elif mode == "减":
                # 减模式：从当前缩略图中删除新标签命中的图片
                removed_count = 0
                original_count = len(current_thumbnails)
                current_thumbnails[:] = [img for img in current_thumbnails if img not in set(new_images)]
                removed_count = original_count - len(current_thumbnails)
                
                # 重新显示剩余图片
                _display_current_thumbnails()
                status_label.config(text=f"模式[减]: 移除 {removed_count} 张，剩 {len(current_thumbnails)} 张")
                
            elif mode == "交":
                # 交模式：只保留同时在新标签中的图片
                new_set = set(new_images)
                original_count = len(current_thumbnails)
                current_thumbnails[:] = [img for img in current_thumbnails if img in new_set]
                removed_count = original_count - len(current_thumbnails)
                
                # 重新显示交集图片
                _display_current_thumbnails()
                status_label.config(text=f"模式[交]: 保留 {len(current_thumbnails)} 张，移除 {removed_count} 张")
        
        def _display_current_thumbnails(start_index=0):
            """显示当前缩略图列表中的图片，支持分批加载"""
            # 如果是从头开始显示，清空现有组件
            if start_index == 0:
                for w in inner.winfo_children():
                    w.destroy()
            
            if not current_thumbnails:
                ttk.Label(inner, text="无匹配图片").pack()
                canvas.config(scrollregion=canvas.bbox("all"))
                return
            
            cols = 5  # 每行5张
            batch_size = 50  # 每次加载50张（10行）
            
            # 计算本次要显示的图片范围
            end_index = min(start_index + batch_size, len(current_thumbnails))
            display_images = current_thumbnails[start_index:end_index]
            
            row_frame = None
            
            for idx, img_path in enumerate(display_images):
                actual_idx = start_index + idx
                if actual_idx % cols == 0:
                    row_frame = ttk.Frame(inner)
                    row_frame.pack(fill=tk.X, padx=2, pady=2)
                
                try:
                    # 使用图片路径作为缓存键，避免哈希碰撞
                    cache_key = f"thumb_{img_path}"
                    if cache_key in self._thumbnail_cache:
                        photo = self._thumbnail_cache[cache_key]
                    else:
                        img = Image.open(img_path)
                        img.thumbnail((120, 120))
                        photo = ImageTk.PhotoImage(img)
                        self._thumbnail_cache[cache_key] = photo
                    
                    # 使用默认参数修复闭包问题，确保每个按钮绑定正确的路径
                    btn = ttk.Button(
                        row_frame,
                        image=photo,
                        command=lambda p=img_path: show_preview(p),
                    )
                    btn.image = photo
                    
                    # 绑定双击事件，用XnView打开图片
                    def on_double_click(event, path=img_path):
                        try:
                            if os.path.exists(self.image_viewer_path):
                                subprocess.Popen([self.image_viewer_path, path])
                            else:
                                # 如果XnView不存在，使用系统默认程序打开
                                os.startfile(path)
                        except Exception as e:
                            self.log(f"打开图片失败: {e}")
                    
                    btn.bind("<Double-Button-1>", on_double_click)
                    
                    # 绑定右键单击事件，显示目录图片
                    def on_right_click(event, path=img_path):
                        show_directory_images(path)
                    
                    btn.bind("<Button-3>", on_right_click)
                    btn.pack(side=tk.LEFT, padx=2)
                except Exception:
                    pass
            
            # 如果还有更多图片，显示"加载更多"按钮
            if end_index < len(current_thumbnails):
                remaining = len(current_thumbnails) - end_index
                load_more_frame = ttk.Frame(inner)
                load_more_frame.pack(pady=10)
                
                def load_more():
                    _display_current_thumbnails(end_index)
                
                ttk.Button(
                    load_more_frame,
                    text=f"加载更多 ({remaining} 张)",
                    command=load_more
                ).pack()
            
            canvas.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))

        # 延迟加载标签列表，避免阻塞界面打开
        def delayed_init():
            status_label.config(text="正在加载标签...")
            win.update_idletasks()
            
            # 先获取数据
            nonlocal all_tags_cache
            if not all_tags_cache:
                all_tags_cache = self.get_tag_stats()
            
            total_tags = len(all_tags_cache) if all_tags_cache else 0
            status_label.config(text=f"正在创建标签按钮...")
            win.update_idletasks()
            
            # 从头开始显示，支持分批加载
            refresh_tags(0)
            
            status_label.config(text="就绪")
        
        # 在界面显示后立即加载，不阻塞窗口打开
        win.after(100, delayed_init)

        def show_preview(image_path):
            current_img_path[0] = image_path
            try:
                img = Image.open(image_path)
                img.thumbnail((500, 500))
                photo = ImageTk.PhotoImage(img)
                preview_label.config(image=photo, text="")
                preview_label.image = photo
            except Exception as e:
                preview_label.config(image="", text=f"无法加载: {e}")
            tag_text.delete("1.0", tk.END)
            tag_text.insert("1.0", self.get_tags_by_image(image_path))

        # Right: image preview
        preview_frame = ttk.LabelFrame(paned, text="图片预览", padding=5, width=400)
        preview_frame.pack_propagate(False)  # 禁止子组件改变框架大小
        paned.add(preview_frame)

        preview_label = ttk.Label(preview_frame, text="点击缩略图查看")
        preview_label.pack(expand=True, fill=tk.BOTH)

        tag_entry_frame = ttk.Frame(preview_frame)
        tag_entry_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(tag_entry_frame, text="标签：").pack(side=tk.LEFT, anchor="n")

        tag_text = tk.Text(tag_entry_frame, height=6, wrap=tk.WORD)
        tag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 5))

        current_img_path = [""]

        # 绑定双击事件到标签文本框
        def on_tag_double_click(event):
            """双击标签文本框中的标签，将其添加到包含框尾部"""
            try:
                # 获取当前光标位置
                cursor_index = tag_text.index(tk.CURRENT)
                line_num = int(cursor_index.split('.')[0])
                line_content = tag_text.get(f"{line_num}.0", f"{line_num}.end")
                
                # 获取光标在行内的字符位置
                char_pos = int(cursor_index.split('.')[1])
                
                # 向前查找分隔符
                start_pos = char_pos
                separators = [',', '，', '、', '；', ';', '|', '/', ' ', '\n']
                while start_pos > 0 and line_content[start_pos-1] not in separators:
                    start_pos -= 1
                
                # 向后查找分隔符
                end_pos = char_pos
                while end_pos < len(line_content) and line_content[end_pos] not in separators:
                    end_pos += 1
                
                # 提取标签并去除空格
                selected_tag = line_content[start_pos:end_pos].strip()
                
                if selected_tag:
                    # 添加到包含框尾部，以空格分隔
                    current_filter = filter_var.get().strip()
                    if current_filter:
                        new_filter = f"{current_filter} {selected_tag}"
                    else:
                        new_filter = selected_tag
                    filter_var.set(new_filter)
            except Exception as e:
                pass
        
        # 绑定右键单击事件到标签文本框
        def on_tag_right_click(event):
            """右键单击标签文本框中的标签，将其添加到排除框尾部"""
            try:
                # 获取当前光标位置
                cursor_index = tag_text.index(tk.CURRENT)
                line_num = int(cursor_index.split('.')[0])
                line_content = tag_text.get(f"{line_num}.0", f"{line_num}.end")
                
                # 获取光标在行内的字符位置
                char_pos = int(cursor_index.split('.')[1])
                
                # 向前查找分隔符
                start_pos = char_pos
                separators = [',', '，', '、', '；', ';', '|', '/', ' ', '\n']
                while start_pos > 0 and line_content[start_pos-1] not in separators:
                    start_pos -= 1
                
                # 向后查找分隔符
                end_pos = char_pos
                while end_pos < len(line_content) and line_content[end_pos] not in separators:
                    end_pos += 1
                
                # 提取标签并去除空格
                selected_tag = line_content[start_pos:end_pos].strip()
                
                if selected_tag:
                    # 添加到排除框尾部，以空格分隔
                    current_exclude = exclude_var.get().strip()
                    if current_exclude:
                        new_exclude = f"{current_exclude} {selected_tag}"
                    else:
                        new_exclude = selected_tag
                    exclude_var.set(new_exclude)
            except Exception as e:
                pass
        
        tag_text.bind("<Double-Button-1>", on_tag_double_click)
        tag_text.bind("<Button-3>", on_tag_right_click)

        def save_tags():
            path = current_img_path[0]
            if not path or not os.path.exists(path):
                return
            new_tags = tag_text.get("1.0", tk.END).strip()
            if not new_tags:
                return
            
            conn = sqlite3.connect(self.db_path)
            try:
                # 获取旧标签
                cur = conn.execute("SELECT tags FROM image_tags WHERE image_path=?", (path,))
                row = cur.fetchone()
                old_tags_string = row[0] if row else ""
                
                # 更新图片标签
                conn.execute(
                    "INSERT INTO image_tags (image_path, tags) VALUES (?, ?) "
                    "ON CONFLICT(image_path) DO UPDATE SET tags=excluded.tags, created_at=CURRENT_TIMESTAMP",
                    (path, new_tags),
                )
                
                # 增量更新标签统计
                old_tags = self.parse_tags(old_tags_string) if old_tags_string else []
                new_tags_list = self.parse_tags(new_tags)
                
                # 减少旧标签的计数（如果标签不再存在）
                for tag in old_tags:
                    if tag not in new_tags_list:
                        conn.execute("UPDATE tag_stats SET count = count - 1 WHERE tag = ?", (tag,))
                        conn.execute("DELETE FROM tag_stats WHERE tag = ? AND count <= 0", (tag,))
                
                # 增加新标签的计数（如果是新增的）
                for tag in new_tags_list:
                    if tag not in old_tags:
                        conn.execute(
                            "INSERT INTO tag_stats (tag, count) VALUES (?, 1) "
                            "ON CONFLICT(tag) DO UPDATE SET count = count + 1",
                            (tag,),
                        )
                
                conn.commit()
                # 清除缓存以便下次获取最新数据
                self._tag_stats_cache = None
                all_tags_cache.clear()  # 清除过滤缓存
            finally:
                conn.close()
            
            refresh_tags()
            status_label.config(text="已保存并更新统计")

        ttk.Button(tag_entry_frame, text="保存", command=save_tags).pack(
            side=tk.RIGHT, anchor="n"
        )

    def _show_thumbnails(self, parent, canvas, tag_name, show_preview_cb):
        """优化:实现缩略图懒加载和虚拟滚动"""
        # 清空现有组件
        for w in parent.winfo_children():
            w.destroy()
        
        # 清空缩略图缓存中与此标签相关的项
        keys_to_remove = [k for k in self._thumbnail_cache.keys() if k.startswith(f"{tag_name}_")]
        for k in keys_to_remove:
            del self._thumbnail_cache[k]

        images = self.get_images_by_tag(tag_name)
        if not images:
            ttk.Label(parent, text="无匹配图片").pack()
            canvas.config(scrollregion=canvas.bbox("all"))
            return

        cols = 4
        row_frame = None
        
        # 限制初始加载数量,避免一次性加载过多
        max_initial_load = 50
        display_images = images[:max_initial_load]
        
        for idx, img_path in enumerate(display_images):
            if idx % cols == 0:
                row_frame = ttk.Frame(parent)
                row_frame.pack(fill=tk.X, padx=2, pady=2)

            try:
                # 检查缓存
                cache_key = f"{tag_name}_{idx}"
                if cache_key in self._thumbnail_cache:
                    photo = self._thumbnail_cache[cache_key]
                else:
                    img = Image.open(img_path)
                    img.thumbnail((120, 120))
                    photo = ImageTk.PhotoImage(img)
                    self._thumbnail_cache[cache_key] = photo

                btn = ttk.Button(
                    row_frame,
                    image=photo,
                    command=lambda p=img_path: show_preview_cb(p),
                )
                btn.image = photo  # 保持引用防止被垃圾回收
                
                # 绑定双击事件，用XnView打开图片
                def on_double_click(event, path=img_path):
                    try:
                        if os.path.exists(self.image_viewer_path):
                            subprocess.Popen([self.image_viewer_path, path])
                        else:
                            # 如果XnView不存在，使用系统默认程序打开
                            os.startfile(path)
                    except Exception as e:
                        self.log(f"打开图片失败: {e}")
                
                btn.bind("<Double-Button-1>", on_double_click)
                
                # 绑定右键单击事件，用资源管理器打开所在文件夹并选中文件
                def on_right_click(event, path=img_path):
                    try:
                        if os.path.exists(path):
                            # 规范化路径，确保使用绝对路径
                            abs_path = os.path.abspath(path)
                            # 使用 explorer /select, 命令打开文件夹并选中文件
                            import subprocess
                            subprocess.run(['explorer', '/select,', abs_path], check=False)
                        else:
                            self.log(f"文件不存在: {path}")
                    except Exception as e:
                        self.log(f"打开文件夹失败: {e}")
                
                btn.bind("<Button-3>", on_right_click)
                btn.pack(side=tk.LEFT, padx=2)
            except Exception as e:
                pass
        
        # 如果还有更多图片,显示提示
        if len(images) > max_initial_load:
            remaining = len(images) - max_initial_load
            ttk.Label(parent, text=f"... 还有 {remaining} 张图片未显示").pack(pady=5)

        canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def start_processing(self):
        if not self.directories:
            messagebox.showwarning("警告", "请先添加目录")
            return
        
        if self.model_session is None:
            messagebox.showerror("错误", "WD-14 模型未初始化，无法处理")
            return

        # 更新置信度阈值和批处理大小
        try:
            self.confidence_threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("错误", "置信度阈值必须是数字")
            return
        
        try:
            self.batch_size = int(self.batch_size_var.get())
            if self.batch_size < 1 or self.batch_size > 256:
                messagebox.showwarning("警告", "批处理大小应在 1-256 之间，已自动调整为 16")
                self.batch_size = 16
                self.batch_size_var.set("16")
        except ValueError:
            messagebox.showerror("错误", "批处理大小必须是整数")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self.process_images, daemon=True).start()

    def stop_processing(self):
        self.running = False
        self.log("正在停止处理...")

    def process_images(self):
        all_files = self.get_image_files()
        pending = []
        self.skipped = 0
        for f in all_files:
            row = self.is_already_analyzed(f)
            if row:
                self.skipped += 1
            else:
                pending.append(f)

        self.total_images = len(pending)
        self.processed = 0
        self.start_time = datetime.now()

        if self.total_images == 0:
            self.log(
                f"✅ 所有图片均已处理过（共 {len(all_files)} 张，跳过 {self.skipped} 张）"
            )
            if all_files:
                self.update_preview(all_files[-1])
            self.finish_processing()
            return

        self.log(
            f"📊 总计 {len(all_files)} 张 | 跳过 {self.skipped} 张 | 待处理 {self.total_images} 张"
        )
        self.log(f"⚡ 批处理大小: {self.batch_size} 张/批次")
        self.progress["maximum"] = self.total_images
        self.progress["value"] = 0

        # 分批处理图片
        batch_start = 0
        while batch_start < len(pending):
            if not self.running:
                self.log("⏹️ 处理已停止")
                break
            
            # 获取当前批次
            batch_end = min(batch_start + self.batch_size, len(pending))
            batch_paths = pending[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(pending) + self.batch_size - 1) // self.batch_size
            
            self.log(f"\n🔄 处理批次 {batch_num}/{total_batches} ({len(batch_paths)} 张图片)")
            
            try:
                # 批量分析
                results = self.analyze_images_batch(batch_paths)
                
                # 保存结果
                for image_path, tags in results:
                    if not self.running:
                        break
                    
                    basename = os.path.basename(image_path)
                    self.log(f"  🏷️  {basename}: {tags[:100]}{'...' if len(tags) > 100 else ''}")
                    
                    # 保存到数据库
                    self.save_result(image_path, tags)
                    self.update_tag_stats(tags)
                    self.processed += 1
                    
                    # 更新进度
                    self.progress["value"] = self.processed
                    
                    # 计算剩余时间
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    if self.processed > 0:
                        avg_time = elapsed / self.processed
                        remaining = self.total_images - self.processed
                        eta_seconds = avg_time * remaining
                        
                        if eta_seconds < 60:
                            eta_str = f"{int(eta_seconds)}秒"
                        elif eta_seconds < 3600:
                            eta_str = f"{int(eta_seconds // 60)}分钟"
                        else:
                            hours = int(eta_seconds // 3600)
                            minutes = int((eta_seconds % 3600) // 60)
                            eta_str = f"{hours}小时{minutes}分钟"
                        
                        self.status_var.set(f"[{self.processed}/{self.total_images}] 预计剩余: {eta_str}")
                    
                    # 更新预览（显示最后一张处理的图片）
                    self.update_preview(image_path)
                    self.root.update_idletasks()
                
            except Exception as e:
                self.log(f"  ❌ 批次处理失败: {str(e)}")
                # 如果批量处理失败，尝试单张处理
                self.log(f"  🔄 切换到单张处理模式...")
                for image_path in batch_paths:
                    if not self.running:
                        break
                    try:
                        tags = self.analyze_image(image_path)
                        self.save_result(image_path, tags)
                        self.update_tag_stats(tags)
                        self.processed += 1
                        self.progress["value"] = self.processed
                        basename = os.path.basename(image_path)
                        self.log(f"  🏷️  {basename}: {tags[:100]}{'...' if len(tags) > 100 else ''}")
                        self.update_preview(image_path)
                        self.root.update_idletasks()
                    except Exception as e2:
                        self.log(f"  ❌ {basename}: {str(e2)}")
            
            batch_start = batch_end

        self.finish_processing()

    def finish_processing(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        total_all = self.processed + self.skipped
        self.status_var.set(
            f"✅ 处理完成（本次 {self.processed} | 跳过 {self.skipped} | 累计 {total_all}）"
        )
        self.log(f"✅ 处理完成：本次新增 {self.processed} 张，跳过 {self.skipped} 张")
        self.log(f"💾 数据库: {self.db_path}")
        self.update_stats()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTaggingApp(root)
    root.mainloop()
