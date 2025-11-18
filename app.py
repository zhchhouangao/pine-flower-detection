import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# 页面配置
st.set_page_config(
    page_title="Pine Flower Phenology Recognition",
    page_icon="🌲",
    layout="wide"
)
# 🎯 模型文件检查
model_path = 'models/best.pt'
if os.path.exists(model_path):
    st.sidebar.success(f"✅ 模型文件加载成功 ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
else:
    st.sidebar.error("❌ 模型文件未找到")

# 松花时期类别映射
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'display_name': 'Elongation Stage'},
    1: {'name': 'ripening stage', 'color': (0, 165, 255), 'display_name': 'Ripening Stage'},
    2: {'name': 'decline stage', 'color': (0, 0, 255), 'display_name': 'Decline Stage'}
}


class StreamlitDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """加载YOLOv11模型"""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            self.model = None

    def detect_image(self, image):
        """执行图片检测"""
        try:
            st.write("---")
            st.write("🔍 **开始检测过程**")
            st.write(f"📐 输入图像尺寸: {image.shape}")

            if self.model is not None:
                st.write("✅ 使用YOLO模型进行检测...")

                # 执行检测
                results = self.model(image)
                st.write(f"📊 YOLO返回 {len(results)} 个检测结果")

                detections = []
                for i, result in enumerate(results):
                    boxes = result.boxes
                    if boxes is not None:
                        st.write(f"🎯 结果 {i + 1}: 检测到 {len(boxes)} 个目标")

                        for j, box in enumerate(boxes):
                            class_id = int(box.cls.item())
                            confidence = box.conf.item()
                            bbox = box.xyxy[0].tolist()

                            st.write(f"   📦 目标 {j + 1}:")
                            st.write(
                                f"     类别: {class_id} ({PINE_FLOWER_CLASSES.get(class_id, {}).get('display_name', 'Unknown')})")
                            st.write(f"     置信度: {confidence:.3f}")
                            st.write(f"     位置: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

                            class_info = PINE_FLOWER_CLASSES.get(class_id, {
                                'name': 'unknown', 'color': (255, 255, 255), 'display_name': 'Unknown Stage'
                            })

                            detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'class_name': class_info['name'],
                                'display_name': class_info['display_name'],
                                'class_id': class_id,
                                'color': class_info['color']
                            })
                    else:
                        st.warning(f"⚠️ 结果 {i + 1}: 未检测到任何目标")

                st.write(f"🎉 总共检测到: {len(detections)} 个松花")

            else:
                st.warning("⚠️ 模型未加载，使用模拟检测")
                detections = self.mock_detect(image)

            # 绘制检测结果
            st.write("🖌️ 开始绘制检测框...")
            result_image = self.draw_detections(image.copy(), detections)
            return detections, result_image

        except Exception as e:
            st.error(f"❌ 检测过程中出错: {e}")
            import traceback
            st.error("错误详情:")
            st.code(traceback.format_exc())
            return self.mock_detect(image), image

    def detect_video(self, video_path):
        """执行视频检测"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Cannot open video file")
                return [], None

            # 创建临时输出文件
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_detections = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 更新进度
                if frame_count % 10 == 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

                # 每5帧检测一次
                if frame_count % 5 == 0:
                    if self.model is not None:
                        results = self.model(frame)
                        frame_detections = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls.item())
                                class_info = PINE_FLOWER_CLASSES.get(class_id, {
                                    'name': 'unknown', 'color': (255, 255, 255), 'display_name': 'Unknown Stage'
                                })

                                frame_detections.append({
                                    'bbox': box.xyxy[0].tolist(),
                                    'confidence': box.conf.item(),
                                    'class_name': class_info['name'],
                                    'display_name': class_info['display_name'],
                                    'class_id': class_id,
                                    'color': class_info['color']
                                })
                    else:
                        frame_detections = self.mock_detect(frame)

                    video_detections.extend(frame_detections)

                # 绘制检测框
                result_frame = self.draw_detections(frame.copy(), frame_detections if frame_count % 5 == 0 else [])
                out.write(result_frame)
                frame_count += 1

            cap.release()
            out.release()
            progress_bar.progress(1.0)
            status_text.text("Processing completed!")

            return video_detections, output_path

        except Exception as e:
            st.error(f"Video processing failed: {e}")
            return [], None

    def mock_detect(self, image):
        """模拟检测"""
        height, width = image.shape[:2]
        detections = []
        import random
        num_detections = random.randint(2, 4)

        for i in range(num_detections):
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(80, 200)
            confidence = round(0.7 + random.random() * 0.25, 2)
            class_id = random.randint(0, 2)
            class_info = PINE_FLOWER_CLASSES[class_id]

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_info['name'],
                'display_name': class_info['display_name'],
                'class_id': class_id,
                'color': class_info['color']
            })
        return detections

    def draw_detections(self, image, detections):
        """绘制检测框"""
        st.write(f"🖌️ 需要绘制 {len(detections)} 个检测框")

        if len(detections) == 0:
            st.warning("⚠️ 没有检测框需要绘制，返回原图")
            return image

        # 获取图像尺寸
        image_height, image_width = image.shape[:2]
        st.write(f"📏 画布尺寸: 宽={image_width}, 高={image_height}")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            display_name = det['display_name']

            st.write(f"  🎨 绘制第 {i + 1} 个框: {display_name}")
            st.write(f"     置信度: {conf:.2f}")
            st.write(f"     坐标: [{x1}, {y1}, {x2}, {y2}]")

            # 检查坐标是否合理
            if x1 >= x2 or y1 >= y2:
                st.error(f"     ❌ 坐标无效: x1>=x2 或 y1>=y2")
                continue

            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                st.warning(f"     ⚠️ 坐标部分超出图像范围")

            # 画检测框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            st.write(f"     ✅ 已绘制边界框")

            # 画标签背景
            label = f"{display_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # 计算标签位置（确保不超出图像上边界）
            label_bg_y1 = max(y1 - label_size[1] - 10, 0)
            label_bg_y2 = y1
            label_bg_x2 = x1 + label_size[0] + 5

            cv2.rectangle(image, (x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
            st.write(f"     ✅ 已绘制标签背景")

            # 画文字
            text_y = max(y1 - 5, label_size[1] - 5)
            cv2.putText(image, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            st.write(f"     ✅ 已绘制文字标签")

        st.success("🎨 所有检测框绘制完成!")
        return image

    def get_statistics(self, detections):
        """获取统计信息"""
        stats = {'total_count': 0, 'by_stage': defaultdict(int)}
        if not detections:
            return stats

        stats['total_count'] = len(detections)
        for det in detections:
            stage = det['display_name']
            stats['by_stage'][stage] += 1

        return stats


# 初始化检测器
@st.cache_resource
def load_detector():
    return StreamlitDetector('models/best.pt')


def main():
    # 标题
    st.title("🌲 Pine Flower Phenology Recognition System")
    st.markdown("Based on YOLOv11 - Detect elongation, ripening, and decline stages")

    # 侧边栏
    st.sidebar.title("About")
    st.sidebar.info("This system uses YOLOv11 to detect and classify pine flower phenology stages.")

    # 文件上传
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'],
        help="Supported formats: JPG, PNG, MP4, AVI, MOV"
    )

    if uploaded_file is not None:
        # 显示文件信息
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type
        }
        st.write("File details:", file_details)

        # 加载检测器
        detector = load_detector()

        if st.button("Start Detection", type="primary"):
            with st.spinner("Processing..."):
                # 根据文件类型处理
                if uploaded_file.type.startswith('image'):
                    # 图片处理
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 检测
                    detections, result_image = detector.detect_image(image)
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                    # 显示结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image_rgb, use_column_width=True)
                    with col2:
                        st.subheader("Detection Result")
                        st.image(result_image_rgb, use_column_width=True)

                elif uploaded_file.type.startswith('video'):
                    # 视频处理
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # 检测
                    detections, result_path = detector.detect_video(tmp_path)

                    # 显示结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Video")
                        st.video(uploaded_file)
                    with col2:
                        st.subheader("Detection Result")
                        if result_path:
                            with open(result_path, 'rb') as f:
                                st.video(f.read())

                    # 清理临时文件
                    os.unlink(tmp_path)
                    if result_path and os.path.exists(result_path):
                        os.unlink(result_path)

                # 显示统计信息
                st.subheader("📊 Detection Statistics")
                stats = detector.get_statistics(detections)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Detections", stats['total_count'])

                with col2:
                    for stage, count in stats['by_stage'].items():
                        st.metric(f"{stage}", count)

                # 显示检测详情
                st.subheader("🔍 Detection Details")
                if detections:
                    for i, det in enumerate(detections):
                        st.write(
                            f"**Pine Flower {i + 1}**: {det['display_name']} (Confidence: {det['confidence']:.2f})")
                else:
                    st.info("No pine flowers detected")

                st.success(f"Detection completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
