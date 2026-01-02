import os
import re
import cv2
import time
import base64
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from openai import OpenAI
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, openai_api_key: str, yolo_model_path: str = None):
        self.client = OpenAI(api_key=openai_api_key)
        self.gpt_model = "gpt-5"  
        self.yolo_model = self.load_yolo_model(yolo_model_path)
        
    def load_yolo_model(self, model_path: str = None):
        try:
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                return model
            return None
        except:
            return None
    
    def detect_plates(self, image_np: np.ndarray):
        if self.yolo_model is None:
            return [], []
        
        results = self.yolo_model(image_np, conf=0.3, verbose=False)
        
        boxes = []
        cropped_images = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes.cpu().numpy():
                    confidence = box.conf[0]
                    
                    if confidence > 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2, confidence])
                            
                            cropped = image_np[y1:y2, x1:x2]
                            if cropped.size > 0:
                                cropped_images.append(Image.fromarray(cropped))
        
        return boxes, cropped_images
    
    def read_plate_gpt(self, pil_image: Image.Image) -> str:
        try:
            # Convert PIL Image to base64
            import io
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            image_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please read the vehicle license plate from this image. Only return the plate number, nothing else."},
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50
            )
            
            latency = time.time() - start_time
            print(f"GPT-5 API Latency: {latency:.2f} seconds")
            
            text = response.choices[0].message.content.strip()
            
            filter_words = ["sorry", "can't", "cannot", "assist", "help", "i'm", "apologize", 
                          "unable", "unable", "i am", "don't", "doesn't", "won't"]
            
            if any(word in text.lower() for word in filter_words):
                return "NOT_DETECTED"
            

            cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
            
            patterns = [
                r'\b[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}\b', 
                r'\b[A-Z]{1,2}\d{4}[A-Z]{1,3}\b', 
                r'\b\d{4}\s?[A-Z]{1,3}\b',  
                r'\b[A-Z]{1,2}\s?\d{1,4}\b',  
            ]
            
            for pattern in patterns:
                match = re.search(pattern, cleaned_text)
                if match:
                    detected_plate = match.group()
                    # Validasi panjang minimum
                    if len(detected_plate) >= 3:
                        return detected_plate
            
            # Coba pattern yang lebih general
            general_pattern = r'\b[A-Z0-9\s]{3,12}\b'
            match = re.search(general_pattern, cleaned_text)
            if match:
                candidate = match.group().strip()
                # Cek apakah mengandung kombinasi huruf dan angka
                if any(c.isdigit() for c in candidate) and any(c.isalpha() for c in candidate):
                    if 3 <= len(candidate) <= 12:
                        return candidate
            
            return "NOT_DETECTED"
            
        except Exception as e:
            print(f"Error in GPT-5 processing: {str(e)}")
            
            # Fallback ke GPT-4o jika GPT-5 tidak tersedia
            try:
                print("Falling back to GPT-4o...")
                self.gpt_model = "gpt-4o"
                return self.read_plate_gpt(pil_image)
            except:
                return "NOT_DETECTED"
    
    def draw_boxes(self, image_np: np.ndarray, boxes: list, plate_texts: list = None):
        """
        Gambar bounding box dan teks pada gambar
        Hanya tampilkan teks jika bukan "NOT_DETECTED"
        """
        img_with_boxes = image_np.copy()
        
        for i, (x1, y1, x2, y2, confidence) in enumerate(boxes):
            # Gambar bounding box
            color = (0, 255, 0)  # Hijau
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Gambar confidence score
            label = f"Plate {i+1}: {confidence:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Tambahkan teks plat jika ada dan berhasil dideteksi
            if plate_texts and i < len(plate_texts):
                plate_text = plate_texts[i]
                if plate_text != "NOT_DETECTED":
                    cv2.putText(img_with_boxes, f"{plate_text}", 
                               (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return img_with_boxes

def main():
    st.set_page_config(
        page_title="License Plate Recognition System",
        page_icon="🚗",
        layout="wide"
    )
    
    # Sidebar untuk konfigurasi
    st.sidebar.title("⚙️ Configuration")
    
    # Input API Key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("api_key", "")
    )
    
    if api_key:
        st.session_state.api_key = api_key
    
    # Pilih model YOLO
    yolo_model_path = st.sidebar.text_input(
        "YOLO Model Path (optional)",
        value="bestyolov11.pt"
    )
    
    # Initialize detector
    detector = None
    if api_key:
        try:
            detector = LicensePlateDetector(api_key, yolo_model_path)
            st.sidebar.success("✅ Detector initialized successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing detector: {e}")
    
    # Tab untuk memilih mode - HANYA DUA TAB
    tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Video Upload"])
    
    # Tab 1: Image Upload
    with tab1:
        st.header("Image License Plate Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="image_uploader"
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                image_np = np.array(image.convert('RGB'))
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                st.image(image, caption="Original Image")
                
                if st.button("🔍 Detect License Plates with GPT-5", key="detect_image"):
                    if detector:
                        with st.spinner("Detecting license plates..."):
                            boxes, cropped_images = detector.detect_plates(image_np)
                            
                            if boxes:
                                st.success(f"✅ Found {len(boxes)} license plate(s)")
                                
                                plate_texts = []
                                valid_detections = []
                                valid_crops = []
                                latencies = []
                                
                                progress_bar = st.progress(0)
                                for i, crop in enumerate(cropped_images):
                                    with st.spinner(f"Reading plate {i+1} with GPT-5..."):
                                        start_time = time.time()
                                        crop_resized = crop.resize((400, 150), Image.Resampling.LANCZOS)
                                        plate_text = detector.read_plate_gpt(crop_resized)
                                        latency = time.time() - start_time
                                        latencies.append(latency)
                                        
                                        if plate_text != "NOT_DETECTED":
                                            plate_texts.append(plate_text)
                                            valid_detections.append(boxes[i])
                                            valid_crops.append(crop)
                                            st.success(f"✅ Plate {i+1}: {plate_text} ({latency:.2f}s)")
                                        else:
                                            plate_texts.append("")
                                            st.info(f"Plate {i+1}: Not detected")
                                        
                                        # Update progress
                                        progress = (i + 1) / len(cropped_images)
                                        progress_bar.progress(progress)
                                
                                image_with_boxes = detector.draw_boxes(image_bgr, boxes, plate_texts)
                                image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                                
                                st.image(image_with_boxes_rgb, caption="Detection Results")
                                
                                if valid_detections:
                                    st.subheader("📊 Detection Results")
                                    results_data = []
                                    for i, (box, plate_text, latency) in enumerate(zip(valid_detections, plate_texts, latencies)):
                                        if plate_text != "":
                                            x1, y1, x2, y2, confidence = box
                                            results_data.append({
                                                "Plate #": i+1,
                                                "Confidence": f"{confidence:.2%}",
                                                "Detected Text": plate_text,
                                                "Latency (s)": f"{latency:.2f}",
                                                "Bounding Box": f"({x1}, {y1}) - ({x2}, {y2})"
                                            })
                                    
                                    if results_data:
                                        df = pd.DataFrame(results_data)
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Show average latency
                                        if latencies:
                                            avg_latency = np.mean([l for l, pt in zip(latencies, plate_texts) if pt != ""])
                                            st.info(f"📊 Average GPT-5 processing time: {avg_latency:.2f} seconds")
                                        
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Results as CSV",
                                            data=csv,
                                            file_name="license_plate_results.csv",
                                            mime="text/csv"
                                        )
                                        
                                        st.subheader("🔍 Detected License Plates")
                                        cols = st.columns(min(3, len(valid_crops)))
                                        for i, (crop, plate_text) in enumerate(zip(valid_crops, plate_texts)):
                                            if plate_text != "":
                                                with cols[i % 3]:
                                                    st.image(crop, caption=f"Plate {i+1}: {plate_text}")
                                    else:
                                        st.warning("⚠️ No license plate text could be read from the detected plates.")
                                else:
                                    st.warning("⚠️ License plates detected but text could not be read.")
                            else:
                                st.warning("⚠️ No license plates detected in the image.")
                    else:
                        st.error("Please provide an OpenAI API key in the sidebar.")
        
        with col2:
            st.info("""
            **GPT-5 Vision Powered Detection:**
            
            **Features:**
            - Advanced AI recognition with GPT-5
            - Higher accuracy for difficult plates
            - Better handling of various plate formats
            - Multi-language support
            
            **Instructions:**
            1. Upload an image containing a vehicle
            2. Click 'Detect License Plates with GPT-5'
            3. View detected plates with bounding boxes
            4. See OCR results from GPT-5
            
            **Supported formats:** JPG, JPEG, PNG, BMP
            """)
    
    # Tab 2: Video Upload
    with tab2:
        st.header("Video License Plate Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_video = st.file_uploader(
                "Upload a video",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader"
            )
            
            if uploaded_video is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_video.read())
                temp_file.close()
                
                cap = cv2.VideoCapture(temp_file.name)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                st.info(f"Video Info: {width}x{height}, {fps} FPS")
                
                # PERUBAHAN DI SINI: Tambahkan opsi "Every 1 frame"
                detection_mode = st.radio(
                    "Detection Mode",
                    ["Every frame", "Every 5 frames", "Every 10 frames", "Every 20 frames"],
                    horizontal=True,
                    help="Lower frequency = faster processing, higher = more accurate"
                )
                
                # PERUBAHAN DI SINI: Update mode_interval
                mode_interval = {
                    "Every frame": 1,        # Tambah ini
                    "Every 5 frames": 5,
                    "Every 10 frames": 10,
                    "Every 20 frames": 20
                }
                
                # PERUBAHAN DI SINI: Tambahkan peringatan untuk mode setiap frame
                if detection_mode == "Every frame":
                    st.warning("⚠️ Warning: Processing every frame will be significantly slower but provides maximum accuracy!")
                    
                if st.button("🎬 Process Video with GPT-5", key="process_video"):
                    if detector:
                        stframe = st.empty()
                        progress_bar = st.progress(0)
                        
                        # PERUBAHAN DI SINI: Tambahkan status processing info
                        status_text = st.empty()
                        
                        output_path = "output_video.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_count = 0
                        all_detections = []
                        total_latency = 0
                        detection_count = 0
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
                        # PERUBAHAN DI SINI: Hitung estimasi waktu
                        interval = mode_interval[detection_mode]
                        estimated_detections = total_frames // interval if interval > 0 else total_frames
                        
                        if detection_mode == "Every frame":
                            st.info(f"ℹ️ Processing all {total_frames} frames. This may take a while...")
                        else:
                            st.info(f"ℹ️ Processing {estimated_detections} frames out of {total_frames} total frames.")
                        
                        with st.spinner("Processing video with GPT-5..."):
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # PERUBAHAN DI SINI: Update status secara real-time
                                status_text.text(f"Processing frame {frame_count + 1}/{total_frames}...")
                                
                                # Deteksi setiap N frame
                                if frame_count % interval == 0:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    boxes, cropped_images = detector.detect_plates(frame_rgb)
                                    
                                    plate_texts = []
                                    for crop in cropped_images:
                                        start_time = time.time()
                                        crop_resized = crop.resize((400, 150), Image.Resampling.LANCZOS)
                                        plate_text = detector.read_plate_gpt(crop_resized)
                                        latency = time.time() - start_time
                                        
                                        if plate_text != "NOT_DETECTED":
                                            plate_texts.append(plate_text)
                                            total_latency += latency
                                            detection_count += 1
                                        
                                        # Simpan deteksi
                                        for i, (box, plate_text) in enumerate(zip(boxes, plate_texts)):
                                            if plate_text != "NOT_DETECTED":
                                                x1, y1, x2, y2, confidence = box
                                                all_detections.append({
                                                    "Frame": frame_count,
                                                    "Time (s)": frame_count / fps,
                                                    "Confidence": confidence,
                                                    "Plate Text": plate_text,
                                                    "Latency (s)": f"{latency:.2f}",
                                                    "Bounding Box": f"({x1}, {y1}, {x2}, {y2})"
                                                })
                                    
                                    frame_with_boxes = detector.draw_boxes(frame, boxes, plate_texts)
                                else:
                                    frame_with_boxes = frame
                                
                                out.write(frame_with_boxes)
                                
                                # PERUBAHAN DI SINI: Tampilkan preview lebih sering untuk mode setiap frame
                                preview_interval = 10 if interval == 1 else 30
                                if frame_count % preview_interval == 0:
                                    stframe.image(frame_with_boxes, channels="BGR")
                                
                                progress = (frame_count + 1) / total_frames
                                progress_bar.progress(progress)
                                
                                frame_count += 1
                        
                        cap.release()
                        out.release()
                        status_text.text("✅ Processing complete!")
                        
                        st.success(f"✅ Video processing complete! Processed {frame_count} frames.")
                        
                        # PERUBAHAN DI SINI: Tampilkan statistik deteksi
                        st.info(f"""
                        **Processing Statistics:**
                        - Total frames: {frame_count}
                        - Frames analyzed: {estimated_detections}
                        - Successful plate detections: {detection_count}
                        """)
                        
                        if detection_count > 0:
                            avg_latency = total_latency / detection_count
                            st.info(f"📊 Average GPT-5 latency: {avg_latency:.2f} seconds per detection")
                        
                        if all_detections:
                            st.subheader("📊 Video Detection Results")
                            df_video = pd.DataFrame(all_detections)
                            st.dataframe(df_video, use_container_width=True)
                            
                            # PERUBAHAN DI SINI: Tambahkan chart timeline deteksi
                            if len(all_detections) > 1:
                                st.subheader("📈 Detection Timeline")
                                timeline_data = []
                                for det in all_detections:
                                    timeline_data.append({
                                        "Frame": det["Frame"],
                                        "Time (s)": det["Time (s)"],
                                        "Plate": det["Plate Text"]
                                    })
                                
                                if timeline_data:
                                    timeline_df = pd.DataFrame(timeline_data)
                                    st.line_chart(timeline_df.set_index("Frame")["Time (s)"])
                            
                            csv_video = df_video.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Video Results as CSV",
                                data=csv_video,
                                file_name="video_detection_results.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("ℹ️ No license plate text could be read from the video.")
                        
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=file,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                        
                        os.unlink(temp_file.name)
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    
                    else:
                        st.error("Please provide an OpenAI API key in the sidebar.")
        
        with col2:
            st.info("""
            **GPT-5 Video Processing:**
            1. Upload a video file
            2. Choose detection frequency
            3. Click 'Process Video with GPT-5'
            4. View results and download
            
            **Detection Modes:**
            - **Every frame**: Maximum accuracy, analyzes every single frame
            - Every 5 frames: Very accurate, slower
            - Every 10 frames: Balanced speed/accuracy
            - Every 20 frames: Faster processing
            
            **When to use 'Every frame':**
            - Short videos (under 30 seconds)
            - Critical applications where every plate matters
            - Videos with fast-moving vehicles
            - High-value surveillance footage
            
            **Performance Note:**
            - Every frame: Slowest, highest API cost
            - Every 5-20 frames: Recommended for most use cases
            
            **GPT-5 Advantages:**
            - Superior accuracy for difficult plates
            - Better text recognition in low light
            - Handles distorted/angled plates better
            - Multilingual plate recognition
            
            **Supported formats:** MP4, AVI, MOV, MKV
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>License Plate Recognition System • Powered by YOLOv11 and GPT-5 Vision API</p>
            <small>Note: GPT-5 requires API access. If unavailable, system will automatically fallback to GPT-4o.</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()