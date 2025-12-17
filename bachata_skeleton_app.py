import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import requests

# MediaPipe Tasks API Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class BachataSkeletonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🕺 Detector de Esqueletos - Bachata (Versión 2.0)")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.output_path = None
        self.video_writer = None
        
        # MediaPipe Tasks
        self.landmarker = None
        self.model_path = "pose_landmarker_full.task"
        
        # Conexiones del cuerpo (simplificado para bachata)
        self.POSE_CONNECTIONS = frozenset([
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
            (29, 31), (30, 32), (27, 31), (28, 32)
        ])
        
        # Configuración UI
        self.confidence = tk.DoubleVar(value=0.5)
        self.complexity = tk.IntVar(value=1) # No usado directamente en Tasks API igual que en Legacy
        self.save_video = tk.BooleanVar(value=True)
        self.color_esqueleto = tk.StringVar(value="default")
        
        self.create_widgets()
        
        # Descargar modelo al inicio si no existe
        threading.Thread(target=self.check_and_download_model, daemon=True).start()
    
    def check_and_download_model(self):
        """Descargar el modelo de MediaPipe si no existe localmente"""
        if not os.path.exists(self.model_path):
            self.info_label.config(text="⏳ Descargando modelo de IA (primera vez)...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(self.model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    self.info_label.config(text="✅ Modelo descargado correctamente")
                else:
                    self.info_label.config(text=f"❌ Error descarga: {response.status_code}")
                    if os.path.exists(self.model_path): os.remove(self.model_path)
            except Exception as e:
                self.info_label.config(text=f"❌ Error descargando modelo: {e}")
                if os.path.exists(self.model_path): os.remove(self.model_path)
                messagebox.showerror("Error", f"No se pudo descargar el modelo de IA:\n{e}")

    def create_widgets(self):
        # ============ PANEL SUPERIOR ============
        top_frame = tk.Frame(self.root, bg='#1e1e1e', height=80)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        top_frame.pack_propagate(False)
        
        # Título
        title = tk.Label(top_frame, text="🕺 Detector de Esqueletos (MP Tasks)", 
                        font=('Arial', 20, 'bold'), bg='#1e1e1e', fg='#00ff88')
        title.pack(pady=20)
        
        # ============ PANEL IZQUIERDO (Controles) ============
        left_frame = tk.Frame(self.root, bg='#1e1e1e', width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)
        
        # Botón Cargar Video
        tk.Label(left_frame, text="📁 VIDEO", font=('Arial', 12, 'bold'), 
                bg='#1e1e1e', fg='white').pack(pady=(10, 5))
        
        self.btn_load = tk.Button(left_frame, text="Seleccionar Video", 
                                 command=self.load_video, bg='#00ff88', 
                                 fg='black', font=('Arial', 11, 'bold'),
                                 cursor='hand2', relief=tk.FLAT, padx=20, pady=10)
        self.btn_load.pack(pady=5, fill=tk.X, padx=20)
        
        self.video_label = tk.Label(left_frame, text="Ningún video seleccionado", 
                                   bg='#1e1e1e', fg='#888888', wraplength=260)
        self.video_label.pack(pady=5)
        
        # Separador
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # ⚙️ CONFIGURACIÓN
        tk.Label(left_frame, text="⚙️ CONFIGURACIÓN", font=('Arial', 12, 'bold'),
                bg='#1e1e1e', fg='white').pack(pady=(5, 10))
        
        # Confianza de detección
        tk.Label(left_frame, text="Confianza de detección:", 
                bg='#1e1e1e', fg='white').pack(anchor=tk.W, padx=20)
        
        confidence_scale = tk.Scale(left_frame, from_=0.1, to=1.0, resolution=0.1,
                                   orient=tk.HORIZONTAL, variable=self.confidence,
                                   bg='#2b2b2b', fg='white', highlightthickness=0,
                                   troughcolor='#00ff88', length=240)
        confidence_scale.pack(padx=20, pady=5)
         
        # Color del esqueleto
        tk.Label(left_frame, text="Color del esqueleto:", 
                bg='#1e1e1e', fg='white').pack(anchor=tk.W, padx=20, pady=(10, 0))
        
        color_combo = ttk.Combobox(left_frame, textvariable=self.color_esqueleto,
                                  values=['default', 'azul', 'rojo', 'verde', 'amarillo'],
                                  state='readonly', width=28)
        color_combo.pack(padx=20, pady=5)
        
        # Guardar video
        tk.Checkbutton(left_frame, text="Guardar video procesado", 
                      variable=self.save_video, bg='#1e1e1e', fg='white',
                      selectcolor='#2b2b2b', activebackground='#1e1e1e',
                      activeforeground='#00ff88').pack(pady=10)
        
        # Separador
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # ▶️ CONTROLES
        tk.Label(left_frame, text="▶️ CONTROLES", font=('Arial', 12, 'bold'),
                bg='#1e1e1e', fg='white').pack(pady=(5, 10))
        
        # Botón Procesar
        self.btn_process = tk.Button(left_frame, text="▶ Procesar Video", 
                                    command=self.process_video, bg='#00ff88',
                                    fg='black', font=('Arial', 11, 'bold'),
                                    cursor='hand2', relief=tk.FLAT, padx=20, pady=10,
                                    state=tk.DISABLED)
        self.btn_process.pack(pady=5, fill=tk.X, padx=20)
        
        # Botón Pausar
        self.btn_pause = tk.Button(left_frame, text="⏸ Pausar", 
                                   command=self.toggle_pause, bg='#ff9500',
                                   fg='black', font=('Arial', 11, 'bold'),
                                   cursor='hand2', relief=tk.FLAT, padx=20, pady=10,
                                   state=tk.DISABLED)
        self.btn_pause.pack(pady=5, fill=tk.X, padx=20)
        
        # Botón Detener
        self.btn_stop = tk.Button(left_frame, text="⏹ Detener", 
                                 command=self.stop_video, bg='#ff3b30',
                                 fg='white', font=('Arial', 11, 'bold'),
                                 cursor='hand2', relief=tk.FLAT, padx=20, pady=10,
                                 state=tk.DISABLED)
        self.btn_stop.pack(pady=5, fill=tk.X, padx=20)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(left_frame, mode='determinate')
        self.progress.pack(pady=15, fill=tk.X, padx=20)
        
        self.progress_label = tk.Label(left_frame, text="0%", 
                                      bg='#1e1e1e', fg='white')
        self.progress_label.pack()
        
        # ============ PANEL CENTRAL (Video) ============
        center_frame = tk.Frame(self.root, bg='#2b2b2b')
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas para video
        self.canvas = tk.Canvas(center_frame, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Texto inicial
        self.canvas_text = self.canvas.create_text(400, 300, 
                                                   text="Carga un video para comenzar",
                                                   fill='#888888', 
                                                   font=('Arial', 16))
        
        # ============ PANEL INFERIOR (Info) ============
        bottom_frame = tk.Frame(self.root, bg='#1e1e1e', height=60)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        bottom_frame.pack_propagate(False)
        
        self.info_label = tk.Label(bottom_frame, 
                                   text="ℹ️ Listo | MediaPipe Tasks API",
                                   bg='#1e1e1e', fg='#00ff88', font=('Arial', 10))
        self.info_label.pack(pady=20)
    
    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.video_label.config(text=f"📹 {filename}", fg='#00ff88')
            self.btn_process.config(state=tk.NORMAL)
            self.info_label.config(text=f"✅ Video cargado: {filename}")
            
            # Mostrar primer frame
            self.show_first_frame()
    
    def show_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret:
            self.display_frame(frame)
        cap.release()
    
    def process_video(self):
        if not self.video_path:
            messagebox.showwarning("Advertencia", "Por favor, selecciona un video primero")
            return
            
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Modelo no encontrado", "El modelo de IA aún se está descargando o faltante.")
            return

        self.btn_process.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        
        self.is_playing = True
        self.is_paused = False
        threading.Thread(target=self.process_video_thread, daemon=True).start()
    
    def process_video_thread(self):
        # Crear PoseLandmarker
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.CPU # Forzar CPU para estabilidad en Windows
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.confidence.get(),
            min_pose_presence_confidence=self.confidence.get(),
            min_tracking_confidence=self.confidence.get()
        )
        
        try:
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                
                self.cap = cv2.VideoCapture(self.video_path)
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if self.save_video.get():
                    self.output_path = "bachata_esqueleto_output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                
                while self.cap.isOpened() and self.is_playing:
                    if not self.is_paused:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        
                        # Preparar imagen para MediaPipe
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        # Calcular timestamp en ms
                        timestamp_ms = int((frame_count * 1000) / fps)
                        
                        # Detectar
                        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                        
                        # Dibujar (custom)
                        if detection_result.pose_landmarks:
                            frame = self.draw_custom_landmarks(frame, detection_result.pose_landmarks[0])
                        
                        if self.video_writer:
                            self.video_writer.write(frame)
                        
                        self.display_frame(frame)
                        
                        frame_count += 1
                        progress = (frame_count / total_frames) * 100
                        self.progress['value'] = progress
                        self.progress_label.config(text=f"{int(progress)}%")
                        self.info_label.config(text=f"⏳ Procesando... Frame {frame_count}/{total_frames}")

        except Exception as e:
            print(f"Error procesando video: {e}")
            messagebox.showerror("Error", f"Ocurrió un error: {e}")
        finally:
            self.finish_processing()

    def draw_custom_landmarks(self, image, landmarks):
        """Dibuja landmarks y conexiones manualmente sobre la imagen BGR"""
        h, w, _ = image.shape
        color = self.get_skeleton_color()
        
        # Dibujar líneas de conexión
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Verificar visibilidad/presencia si es necesario, 
                # pero en `pose_landmarks` generalmente están todos
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(image, start_point, end_point, color, 2)
        
        # Dibujar puntos
        for idx, landmark in enumerate(landmarks):
            # Filtrar si queremos solo el esqueleto principal (sin cara 0-10)
            if idx < 11 and idx > 0: # Opcional: saltar cara
               continue
               
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, color, -1)
            
        return image

    def get_skeleton_color(self):
        colors = {
            'default': (0, 255, 0),  # BGR: Verde
            'azul': (255, 0, 0),     # BGR: Azul
            'rojo': (0, 0, 255),     # BGR: Rojo
            'verde': (0, 255, 0),
            'amarillo': (0, 255, 255) # BGR: Amarillo
        }
        return colors.get(self.color_esqueleto.get(), (0, 255, 0))
    
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Dimensiones del canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Dimensiones originales del frame
        frame_height, frame_width, _ = frame.shape
        
        if canvas_width > 1 and canvas_height > 1:
            # Calcular escala para mantener relación de aspecto
            scale_width = canvas_width / frame_width
            scale_height = canvas_height / frame_height
            scale = min(scale_width, scale_height)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # Redimensionar
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Calcular posición centrada
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2
            
            self.canvas.delete("all")
            self.canvas.create_image(x_pos, y_pos, image=photo, anchor=tk.NW)
            self.canvas.image = photo
    
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ Reanudar", bg='#00ff88')
            self.info_label.config(text="⏸ Pausado")
        else:
            self.btn_pause.config(text="⏸ Pausar", bg='#ff9500')
            self.info_label.config(text="⏳ Procesando...")
    
    def stop_video(self):
        self.is_playing = False
    
    def finish_processing(self):
        if self.cap: self.cap.release()
        if self.video_writer: self.video_writer.release()
        
        self.btn_process.config(state=tk.NORMAL)
        self.btn_load.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Pausar", bg='#ff9500')
        self.btn_stop.config(state=tk.DISABLED)
        
        self.is_paused = False
        
        if self.save_video.get() and self.output_path and os.path.exists(self.output_path):
            self.info_label.config(text=f"✅ Video guardado: {self.output_path}")
            messagebox.showinfo("Éxito", f"Video procesado guardado en:\n{self.output_path}")
        else:
            self.info_label.config(text="✅ Procesamiento finalizado (o detenido)")

if __name__ == "__main__":
    root = tk.Tk()
    app = BachataSkeletonApp(root)
    root.mainloop()
