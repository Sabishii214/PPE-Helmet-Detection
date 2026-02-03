"""
PPE Detection Analytics System
Optimized code for PPE compliance monitoring
"""

import os
import json
import cv2
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from config import CLASSES, get_latest_model_path, INFERENCE_THRESHOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PPEDetectionAnalytics:
    """Main analytics class for PPE detection"""
    
    def __init__(self, model_path=None, conf_threshold=None):
        self.model = YOLO(model_path or get_latest_model_path('best'))
        
        # Confidence thresholds (Load from Config)
        self.conf_image = INFERENCE_THRESHOLDS['image']
        self.conf_video = INFERENCE_THRESHOLDS['video']
        self.conf_webcam = INFERENCE_THRESHOLDS['webcam']
        
        # If user explicitly provided a custom threshold via CLI
        if conf_threshold is not None and conf_threshold != 0.2: # 0.2 is default argparse value we want to ignore if possible, OR we change argparse default to None
            logger.info(f"Overriding defaults with user-provided confidence: {conf_threshold}")
            self.conf_image = conf_threshold
            self.conf_video = conf_threshold
            self.conf_webcam = conf_threshold
            self.conf_threshold = conf_threshold
        else:
             self.conf_threshold = self.conf_image # Set a reasonable default for fallback logic

        
        # Temporal smoothing buffers
        from collections import deque
        self.temporal_window = 5
        self.head_history = deque(maxlen=self.temporal_window)
        self.helmet_history = deque(maxlen=self.temporal_window)
        
        self.classes = CLASSES
        self.results_log = []
        
        # Generated timestamp for this run
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f'output/run_{self.run_id}')
        
        # Configuration - Unified 'Input' directory for all media
        self.config = {
            'input': {'base_dir': 'Input'},  # Unified input directory
            'output': {
                'base_dir': str(self.run_dir),
                'report_dir': str(self.run_dir / 'reports'), 
                'evidence_dir': str(self.run_dir / 'evidence')
            },
            'processing': {'video_frame_skip': 1, 'max_video_frames': 1000}
        }
        
        # Create output directories for this specific run
        try:
            for key in ['report_dir', 'evidence_dir']:
                Path(self.config['output'][key]).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print("\n" + "!" * 60)
            print("PERMISSION ERROR: Cannot write to 'output' directory.")
            print("This usually happens when 'output' is owned by root (via Docker).")
            print("FIX: Run 'sudo chown -R $USER:$USER output' on your host machine.")
            print("!" * 60 + "\n")
            raise
            
        logger.info(f"Initialized analytics run: {self.run_id}")
        logger.info(f"Output directory: {self.run_dir}")
    
    def _is_valid_detection(self, cls_name, conf, box, frame_area=None):
        """Validate detection using class-specific confidence and box area"""
       
        # 1. Class-specific confidence gates (Lowered from 0.6/0.55 for better recall)
        if cls_name == 'helmet' and conf < 0.35:
            return False
        if cls_name == 'head' and conf < 0.3:
            return False
            
        # 2. Minimum box area check (if frame_area provided)
        if frame_area:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 0.0005 * frame_area:
                return False
                
        return True

    def detect_ppe(self, image_path):
        """Run detection on image and return results"""
        # Load image to get dimensions for area check
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return {'helmet': 0, 'head': 0, 'boxes': [], 'error': True}
            
        h, w = img.shape[:2]
        frame_area = h * w
        
        results = self.model(image_path, conf=self.conf_image, verbose=False)[0]
        # Initialize detections dictionary based on CLASSES
        detections = {cls: 0 for cls in self.classes}
        detections['boxes'] = []
        
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = self.classes[cls_idx]
            conf = float(box.conf[0])
            bbox = list(map(int, box.xyxy[0]))
            
            if self._is_valid_detection(cls_name, conf, bbox, frame_area):
                detections[cls_name] += 1
                detections['boxes'].append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': bbox
                })
        
        return detections
    
    def check_compliance(self, detections, is_persistent_check=False, head_persistent=False, helmet_persistent=False):

        h_count = detections.get('helmet', 0)
        hd_count = detections.get('head', 0)
        
        if is_persistent_check:
            is_compliant = not head_persistent or helmet_persistent
            
            if not is_compliant:
                violation = 'Persistent Head Detected without Helmet'
            elif helmet_persistent:
                violation = 'Compliant'
            else:
                violation = 'Compliant' # No head persistent
                
        else:
            
            if hd_count > 0:
                is_compliant = False
                violation = f'{hd_count} worker(s) without helmet'
            else:
                is_compliant = True
                violation = 'Compliant'
        
        return {
            'compliant': is_compliant,
            'helmet_count': h_count,
            'head_count': hd_count,
            'violation_type': violation,
            'severity': 'HIGH' if not is_compliant else 'NONE'
        }
    
    def process_image(self, image_path, save_annotated=True):
        """Process single image and optionally save annotated version"""
        detections = self.detect_ppe(image_path)
        
        # Handle load error
        if detections.get('error'):
            return None
            
        compliance = self.check_compliance(detections)
        
        result = {
            'type': 'image',
            'filename': Path(image_path).name,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'compliance': compliance
        }
        
        self.results_log.append(result)
        
        # Save annotated image if enabled
        if save_annotated and detections['boxes']:
            self._save_annotated_image(image_path, detections)
        
        return result
    
    def process_images(self, image_dir=None):
        """Process all images in directory (recursively) using batch inference for better performance"""
        img_dir = Path(image_dir or self.config['input']['base_dir'])
        
        # Recursive search for images in all subdirectories
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(img_dir.rglob(ext)))
        
        if not images:
            logger.warning(f"No images found in {img_dir} (searched recursively)")
            return []
        
        logger.info(f"Processing {len(images)} images from {img_dir} (including subfolders)...")
        results = []
        
        for idx, img_path in enumerate(images, 1):
            res = self.process_image(img_path)
            if res:
                results.append(res)
            if idx % 10 == 0:
                logger.info(f"  {idx}/{len(images)} processed")
        
        logger.info(f"Completed {len(images)} images")
        return results
    
    def process_single_video(self, video_path, save_annotated=True, frame_skip=None, show_live=False):
        """Process single video file frame by frame using in-memory processing """
        # Handle both file paths and camera indices
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path)
            is_camera = True
            source_name = f"Camera {video_path}"
        elif str(video_path).startswith(('http://', 'https://')):
            cap = cv2.VideoCapture(str(video_path))
            is_camera = True
            source_name = "RemoteStream"
        else:
            cap = cv2.VideoCapture(str(video_path))
            is_camera = False
            source_name = Path(video_path).name
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_camera else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else float('inf')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = width * height
        
        # Determine frame skip
        skip = frame_skip if frame_skip is not None else self.config['processing']['video_frame_skip']
        max_frames = self.config['processing']['max_video_frames']
        
        frames_to_process = min(total_frames, max_frames) if not is_camera else max_frames
        logger.info(f"  Processing {source_name}: {frames_to_process} frames (skip={skip})")
        
        frame_results = []
        annotated_frames = []  # Store frames with annotations for video output
        frame_num = 0
        
        # determine confidence based on source
        conf_thresh = self.conf_webcam if is_camera else self.conf_video
        
        # Clear histories for new video
        self.head_history.clear()
        self.helmet_history.clear()
        
        while frame_num < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame - pass frame directly to model (in-memory)
            if frame_num % skip == 0:
                # Preprocessing for Webcam
                inference_frame = frame.copy()
                if is_camera:
                    # Blur stabilization
                    inference_frame = cv2.GaussianBlur(inference_frame, (3, 3), 0)
                    # Brightness normalization (alpha=1.1, beta=10)
                    inference_frame = cv2.convertScaleAbs(inference_frame, alpha=1.1, beta=10)
                
                # Convert BGR to RGB for YOLO
                frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                
                # Run detection directly on numpy array
                results = self.model(frame_rgb, conf=conf_thresh, verbose=False)[0]
                
                # Extract detections
                detections = {cls: 0 for cls in self.classes}
                detections['boxes'] = []
                
                for box in results.boxes:
                    cls_idx = int(box.cls[0])
                    cls_name = self.classes[cls_idx]
                    conf = float(box.conf[0])
                    bbox = list(map(int, box.xyxy[0]))
                    
                    if self._is_valid_detection(cls_name, conf, bbox, frame_area):
                        detections[cls_name] += 1
                        detections['boxes'].append({
                            'class': cls_name,
                            'confidence': conf,
                            'bbox': bbox
                        })
                
                # Temporal Smoothing Logic
                self.head_history.append(detections['head'] > 0)
                self.helmet_history.append(detections['helmet'] > 0)
                
                head_persistent = sum(self.head_history) >= 2 # Reduced from 3
                helmet_persistent = sum(self.helmet_history) >= 2 # Reduced from 3
                
                # Check compliance
                # For webcam/video, use persistent logic
                compliance = self.check_compliance(
                    detections, 
                    is_persistent_check=True,  # Always use persistent check for video/webcam
                    head_persistent=head_persistent,
                    helmet_persistent=helmet_persistent
                )
                
                frame_results.append({
                    'frame': frame_num,
                    'timestamp': frame_num / fps,
                    'detections': detections,
                    'compliance': compliance
                })
                
                # Save annotated frame if enabled (Stable Visualization)
                if save_annotated:
                    # Create copy for manual annotation
                    # results.plot() is too flickering, so we draw only confirmed boxes
                    annotated_frame_bgr = frame.copy()
                    
                    for box_data in detections['boxes']:
                        cls = box_data['class']
                        # Only draw if the class is currently 'persistent' (confirmed 3+ times in window)
                        is_confirmed = (cls == 'head' and head_persistent) or \
                                      (cls == 'helmet' and helmet_persistent)
                        
                        if is_confirmed:
                            x1, y1, x2, y2 = box_data['bbox']
                            conf = box_data['confidence']
                            
                            # Choose color based on class
                            color = (0, 255, 0) if cls == 'helmet' else (0, 0, 255)
                            cv2.rectangle(annotated_frame_bgr, (x1, y1), (x2, y2), color, 2)
                            label = f"{cls} {conf:.2f}"
                            cv2.putText(annotated_frame_bgr, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Apply red overlay if violation detected
                    if not compliance['compliant']:
                        annotated_frame_bgr = self._apply_red_overlay(annotated_frame_bgr, compliance)
                        
                    annotated_frames.append(annotated_frame_bgr)

                    # Live Display
                    if show_live:
                        cv2.imshow(f"PPE Compliance - {source_name}", annotated_frame_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User interrupted live preview.")
                            break
            
            frame_num += 1
        
        cap.release()
        if show_live:
            cv2.destroyAllWindows()
        
        # Calculate video statistics
        violations = [f for f in frame_results if not f['compliance']['compliant']]
        compliance_rate = ((len(frame_results) - len(violations)) / len(frame_results) * 100) if frame_results else 0
        
        # Save annotated video if enabled and frames were collected
        output_video_path = None
        if save_annotated and annotated_frames:
            output_video_path = self._save_annotated_video(
                video_path if not is_camera else f"camera_{video_path}", 
                annotated_frames, fps, width, height
            )
        
        result = {
            'type': 'video' if not is_camera else 'webcam',
            'filename': source_name,
            'timestamp': datetime.now().isoformat(),
            'total_frames': frames_to_process,
            'processed_frames': len(frame_results),
            'violation_frames': len(violations),
            'compliance_rate': compliance_rate,
            'frame_results': frame_results,
            'annotated_video': str(output_video_path) if output_video_path else None
        }
        
        return result
    
    def process_videos(self, video_dir=None, frame_skip=None):
        """Process all videos in directory (recursively)"""
        vid_dir = Path(video_dir or self.config['input']['base_dir'])
        
        # Recursive search for videos in all subdirectories
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
            videos.extend(list(vid_dir.rglob(ext)))
        
        if not videos:
            logger.warning(f"No videos found in {vid_dir} (searched recursively)")
            return []
        
        logger.info(f"Found {len(videos)} videos in {vid_dir} (including subfolders)")
        results = []
        
        for video_path in videos:
            result = self.process_single_video(video_path, frame_skip=frame_skip)
            if result:
                results.append(result)
                self.results_log.append(result)
        
        return results
    
    def process_webcam(self, camera_index=0, duration_seconds=30, frame_skip=None, show_live=True):
        """Process webcam feed for real-time PPE detection """
        logger.info(f"Starting webcam capture from camera {camera_index} for {duration_seconds}s...")
        
        # Calculate max frames based on duration
        fps = 30  # Assume 30 fps for webcam
        original_max_frames = self.config['processing']['max_video_frames']
        self.config['processing']['max_video_frames'] = duration_seconds * fps
        
        result = self.process_single_video(camera_index, save_annotated=True, frame_skip=frame_skip, show_live=show_live)
        
        # Restore original max_frames
        self.config['processing']['max_video_frames'] = original_max_frames
        
        if result:
            self.results_log.append(result)
        
        return result
    
    def _apply_red_overlay(self, image, compliance):
        """Apply semi-transparent red overlay if there are violations"""
        if compliance['compliant']:
            return image
        
        # Create a red overlay
        overlay = image.copy()
        # BGR format: Red is (0, 0, 255)
        overlay[:] = (0, 0, 255) 
        
        # Blend with original image (alpha=0.3 for 30% opacity)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image

    def _save_annotated_image(self, image_path, detections):
        """Save image with bounding boxes using YOLO's built-in plot method"""
        output_dir = Path(self.config['output']['evidence_dir']) / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference again to get the result object with plot capability
        # Use conf_image to match detection logic
        results = self.model(str(image_path), conf=self.conf_image, verbose=False)[0]
        
        # Note: results.plot() might still show boxes we filtered out by area/class-specific conf
        # but the compliance decision (red overlay) is based on the filtered 'detections'
        annotated_img = results.plot()
        
        # Convert RGB back to BGR 
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
        # Check compliance and apply red overlay if needed
        compliance = self.check_compliance(detections)
        if not compliance['compliant']:
            annotated_img_bgr = self._apply_red_overlay(annotated_img_bgr, compliance)
        
        # Use original filename
        output_path = output_dir / Path(image_path).name
        cv2.imwrite(str(output_path), annotated_img_bgr)
    
    def _save_annotated_video(self, video_path, annotated_frames, fps, width, height):
        """Save annotated video with bounding boxes drawn on frames"""
        output_dir = Path(self.config['output']['evidence_dir']) / 'videos'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use original filename
        output_path = output_dir / Path(video_path).name
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create video writer for {output_path}")
            return None
        
        # Write all annotated frames
        for frame in annotated_frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Saved annotated video: {output_path}")
        return output_path
    
    def generate_report(self):
        """Generate comprehensive analytics report"""
        if not self.results_log:
            return {'error': 'No data processed'}
        
        # Separate image and video results
        image_results = [r for r in self.results_log if r['type'] == 'image']
        video_results = [r for r in self.results_log if r['type'] == 'video']
        
        # Calculate image statistics
        img_compliant = sum(1 for r in image_results if r['compliance']['compliant'])
        img_violations = len(image_results) - img_compliant
        
        # Calculate detection totals (images + peak detections from videos)
        total_helmets = sum(r['detections'].get('helmet', 0) for r in image_results) + \
                        sum(max((fr['detections'].get('helmet', 0) for fr in vr.get('frame_results', [])), default=0) for vr in video_results)
        total_heads = sum(r['detections'].get('head', 0) for r in image_results) + \
                      sum(max((fr['detections'].get('head', 0) for fr in vr.get('frame_results', [])), default=0) for vr in video_results)
        
        # Collect all violations
        violations = [
            {
                'filename': r['filename'],
                'type': r['type'],
                'violation': r['compliance']['violation_type'],
                'severity': r['compliance']['severity']
            }
            for r in self.results_log 
            if r.get('compliance', {}).get('compliant') == False
        ]
        
        return {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_images': len(image_results),
                'total_videos': len(video_results),
                'image_compliant': img_compliant,
                'image_violations': img_violations,
                'image_compliance_rate': (img_compliant / len(image_results) * 100) if image_results else 0,
                'total_helmets': total_helmets,
                'total_heads': total_heads
            },
            'violations': violations,
            'image_results': image_results,
            'video_results': video_results
        }
    
    def save_report(self, filename=None):
        """Save report to JSON file"""
        report = self.generate_report()
        
        if 'error' in report:
            # Silence error logs if called when empty, as main.py handles this
            return None
        
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"
        
        report_path = Path(self.config['output']['report_dir']) / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")
        return str(report_path)
    
    def save_text_report(self, filename=None):
        """Save report to human-readable text file"""
        report = self.generate_report()
        
        if 'error' in report:
            # Silence error logs if called when empty, as main.py handles this
            return None
        
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_report_{timestamp}.txt"
        
        report_path = Path(self.config['output']['report_dir']) / filename
        
        s = report['summary']
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PPE COMPLIANCE ANALYTICS REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {report['generated_at']}\n")
            f.write(f"Run ID: {self.run_id}\n\n")
            
            # Image statistics
            f.write("IMAGE ANALYSIS:\n")
            f.write(f"  Total Images: {s['total_images']}\n")
            f.write(f"  Compliant: {s['image_compliant']}\n")
            f.write(f"  Violations: {s['image_violations']}\n")
            f.write(f"  Compliance Rate: {s['image_compliance_rate']:.2f}%\n\n")
            
            # Video statistics
            if s['total_videos'] > 0:
                f.write(f"VIDEO ANALYSIS:\n")
                f.write(f"  Total Videos: {s['total_videos']}\n\n")
            
            # Detection counts
            f.write("DETECTION SUMMARY:\n")
            f.write(f"  Helmets Detected: {s['total_helmets']}\n")
            f.write(f"  Exposed Heads: {s['total_heads']}\n\n")
            
            # Violations
            if report['violations']:
                f.write("VIOLATIONS:\n")
                for i, v in enumerate(report['violations'], 1):
                    f.write(f"  {i}. {v['filename']}: {v['violation']} (Severity: {v['severity']})\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(f"Output Directory: {self.run_dir}\n")
            f.write(f"Evidence: {self.config['output']['evidence_dir']}\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Text report saved: {report_path}")
        return str(report_path)

    
    def print_summary(self):
        """Print summary statistics to console"""
        report = self.generate_report()
        
        if 'error' in report:
            logger.error(report['error'])
            return
        
        s = report['summary']
        
        logger.info("\n" + "="*60)
        logger.info("PPE COMPLIANCE REPORT")
        logger.info("="*60)
        logger.info(f"Generated: {report['generated_at']}")
        
        # Image statistics
        logger.info(f"\nImages:")
        logger.info(f"  Total: {s['total_images']}")
        logger.info(f"  Compliant: {s['image_compliant']}")
        logger.info(f"  Violations: {s['image_violations']}")
        logger.info(f"  Compliance Rate: {s['image_compliance_rate']:.2f}%")
        
        # Video statistics
        if s['total_videos'] > 0:
            logger.info(f"\nVideos: {s['total_videos']}")
        
        # Detection counts
        logger.info(f"\nDetections:")
        logger.info(f"  Helmets: {s['total_helmets']}")
        logger.info(f"  Exposed Heads: {s['total_heads']}")
        
        
        # Top violations
        if report['violations']:
            logger.info(f"\nTop Violations:")
            for i, v in enumerate(report['violations'][:5], 1):
                logger.info(f"  {i}. {v['filename']}: {v['violation']}")
        
        logger.info("="*60)
    
    def run_complete_pipeline(self, input_dir=None, frame_skip=None):
        """Run complete analytics pipeline processing both images and videos in the input directory"""
        logger.info("="*60)
        logger.info("PPE ANALYTICS PIPELINE")
        logger.info("="*60)
        
        # Use default input dir if none provided
        target_dir = input_dir or self.config['input']['base_dir']
        logger.info(f"Target Directory: {target_dir}")
        
        # Process images
        logger.info("\nStep 1: Processing images...")
        self.process_images(target_dir)
        
        # Process videos
        logger.info("\nStep 2: Processing videos...")
        self.process_videos(target_dir, frame_skip=frame_skip)
        
        # Generate and save report
        logger.info("\nStep 3: Generating report...")
        report_path = self.save_report()
        text_report_path = self.save_text_report()
        
        # Print summary
        logger.info("\nStep 4: Printing summary...")
        self.print_summary()
        
        # detailed report
        logger.info(f"\nJSON Report: {report_path}")
        logger.info(f"\nText Report: {text_report_path}")
        logger.info(f"Evidence: {self.config['output']['evidence_dir']}")
        
        # Fix permissions
        self._fix_permissions(self.run_dir)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)

    def _fix_permissions(self, path):
        """Change ownership of file/directory to match workspace owner"""
        try:
            # Get workspace ownership (assuming /workspace or current dir is mounted)
            # Use '.' to get current directory (workspace root) ownership
            workspace_stat = os.stat('.') 
            uid, gid = workspace_stat.st_uid, workspace_stat.st_gid
            
            # Walk through the path and chown everything
            for root, dirs, files in os.walk(path):
                # Change root of walk
                os.chown(root, uid, gid)
                # Change dirs
                for d in dirs:
                    os.chown(os.path.join(root, d), uid, gid)
                # Change files
                for f in files:
                    os.chown(os.path.join(root, f), uid, gid)
            
            # Also fix the top level path itself if it wasn't covered (though walk does cover root usually)
            os.chown(path, uid, gid)
                
            logger.info(f"Fixed permissions for {path} (chown {uid}:{gid})")
        except Exception as e:
            logger.warning(f"Could not fix permissions: {e}")
    
    def clear_results(self):
        """Clear stored results"""
        self.results_log = []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PPE Detection Analytics System')
    parser.add_argument('--mode', choices=['pipeline', 'webcam'], 
                        default='pipeline', help='Processing mode (default: pipeline)')
    parser.add_argument('--input', help='Input directory for media processing')
    parser.add_argument('--model', help='Path to model weights (overrides latest)')
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold (default: auto-detected per mode)')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip count (process every Nth frame)')
    parser.add_argument('--cam', default='0', help='Camera index or stream URL for webcam (default: 0)')
    parser.add_argument('--duration', type=int, default=30, help='Webcam capture duration (sec)')
    
    args = parser.parse_args()
    
    # Handle numeric camera index vs URL string
    cam_source = args.cam
    if args.input and args.mode == 'webcam':
        cam_source = args.input
    
    try:
        cam_source = int(cam_source)
    except ValueError:
        pass # Keep as string if it's a URL
        
    analytics = PPEDetectionAnalytics(model_path=args.model, conf_threshold=args.conf)
    
    if args.mode == 'webcam':
        analytics.process_webcam(camera_index=cam_source, duration_seconds=args.duration, frame_skip=args.skip)
        analytics.save_report()
        analytics.save_text_report()
        analytics.print_summary()
    else: # pipeline
        analytics.run_complete_pipeline(args.input, frame_skip=args.skip)