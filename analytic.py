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
from config import CLASSES, get_latest_model_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PPEDetectionAnalytics:
    """Main analytics class for PPE detection"""
    
    def __init__(self, model_path=None, conf_threshold=0.2):
        self.model = YOLO(model_path or get_latest_model_path('best'))
        self.conf_threshold = conf_threshold
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
        for key in ['report_dir', 'evidence_dir']:
            Path(self.config['output'][key]).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized analytics run: {self.run_id}")
        logger.info(f"Output directory: {self.run_dir}")
    
    def detect_ppe(self, image_path):
        """Run detection on image and return results"""
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)[0]
        detections = {'helmet': 0, 'head': 0, 'person': 0, 'boxes': []}
        
        for box in results.boxes:
            cls = int(box.cls[0])
            detections[self.classes[cls]] += 1
            detections['boxes'].append({
                'class': self.classes[cls],
                'confidence': float(box.conf[0]),
                'bbox': list(map(int, box.xyxy[0]))
            })
        
        return detections
    
    def check_compliance(self, detections):
        """Check if detections meet compliance rules"""
        h_count = detections['helmet']
        hd_count = detections['head']
        p_count = detections['person']
        
        # Rule: No exposed heads and at least one helmet
        is_compliant = (hd_count == 0) and (h_count > 0)
        
        if hd_count > 0:
            violation = f'{hd_count} worker(s) without helmet'
        elif h_count == 0 and p_count > 0:
            violation = 'No helmets detected'
        else:
            violation = 'Compliant'
        
        return {
            'compliant': is_compliant,
            'helmet_count': h_count,
            'head_count': hd_count,
            'person_count': p_count,
            'violation_type': violation,
            'severity': 'HIGH' if hd_count > 0 else 'NONE'
        }
    
    def process_image(self, image_path, save_annotated=True):
        """Process single image and optionally save annotated version"""
        detections = self.detect_ppe(image_path)
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
            results.append(self.process_image(img_path))
            if idx % 10 == 0:
                logger.info(f"  {idx}/{len(images)} processed")
        
        logger.info(f"Completed {len(images)} images")
        return results
    
    def process_single_video(self, video_path, save_annotated=True):
        """Process single video file frame by frame using in-memory processing
        
        Args:
            video_path: Path to video file or integer camera index (0, 1, etc.)
            save_annotated: Whether to save annotated output
        """
        # Handle both file paths and camera indices
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path)
            is_camera = True
            source_name = f"Camera {video_path}"
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
        frame_skip = self.config['processing']['video_frame_skip']
        max_frames = self.config['processing']['max_video_frames']
        
        frames_to_process = min(total_frames, max_frames) if not is_camera else max_frames
        logger.info(f"  Processing {source_name}: {frames_to_process} frames (skip={frame_skip})")
        
        frame_results = []
        annotated_frames = []  # Store frames with annotations for video output
        frame_num = 0
        
        while frame_num < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame - pass frame directly to model (in-memory)
            if frame_num % frame_skip == 0:
                # Convert BGR to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection directly on numpy array
                results = self.model(frame_rgb, conf=self.conf_threshold, verbose=False)[0]
                
                # Extract detections
                detections = {'helmet': 0, 'head': 0, 'person': 0, 'boxes': []}
                for box in results.boxes:
                    cls = int(box.cls[0])
                    detections[self.classes[cls]] += 1
                    detections['boxes'].append({
                        'class': self.classes[cls],
                        'confidence': float(box.conf[0]),
                        'bbox': list(map(int, box.xyxy[0]))
                    })
                
                compliance = self.check_compliance(detections)
                
                frame_results.append({
                    'frame': frame_num,
                    'timestamp': frame_num / fps,
                    'detections': detections,
                    'compliance': compliance
                })
                
                # Save annotated frame if enabled
                if save_annotated:
                    annotated_frame = results.plot()  # Get annotated frame from YOLO
                    # Convert RGB back to BGR for video writing
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    # Apply red overlay if violation detected
                    if not compliance['compliant']:
                        annotated_frame_bgr = self._apply_red_overlay(annotated_frame_bgr, compliance)
                        
                    annotated_frames.append(annotated_frame_bgr)
            
            frame_num += 1
        
        cap.release()
        
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
    
    def process_videos(self, video_dir=None):
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
            result = self.process_single_video(video_path)
            if result:
                results.append(result)
                self.results_log.append(result)
        
        return results
    
    def process_webcam(self, camera_index=0, duration_seconds=30):
        """Process webcam feed for real-time PPE detection
        
        Args:
            camera_index: Camera device index (0 for default camera)
            duration_seconds: How long to capture (in seconds)
        """
        logger.info(f"Starting webcam capture from camera {camera_index} for {duration_seconds}s...")
        
        # Calculate max frames based on duration
        fps = 30  # Assume 30 fps for webcam
        original_max_frames = self.config['processing']['max_video_frames']
        self.config['processing']['max_video_frames'] = duration_seconds * fps
        
        result = self.process_single_video(camera_index, save_annotated=True)
        
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
        results = self.model(str(image_path), conf=self.conf_threshold, verbose=False)[0]
        
        # Use YOLO's built-in plot method for consistent annotations
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
        
        total_helmets = sum(r['detections']['helmet'] for r in image_results)
        total_heads = sum(r['detections']['head'] for r in image_results)
        total_persons = sum(r['detections']['person'] for r in image_results)
        
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
                'total_heads': total_heads,
                'total_persons': total_persons
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
            f.write(f"  Exposed Heads: {s['total_heads']}\n")
            f.write(f"  Persons Detected: {s['total_persons']}\n\n")
            
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
        logger.info(f"  Persons: {s['total_persons']}")
        
        # Top violations
        if report['violations']:
            logger.info(f"\nTop Violations:")
            for i, v in enumerate(report['violations'][:5], 1):
                logger.info(f"  {i}. {v['filename']}: {v['violation']}")
        
        logger.info("="*60)
    
    def run_complete_pipeline(self, image_dir=None, video_dir=None):
        """Run complete analytics pipeline"""
        logger.info("="*60)
        logger.info("PPE ANALYTICS PIPELINE")
        logger.info("="*60)
        
        # Process images
        logger.info("\nStep 1: Processing images...")
        self.process_images(image_dir)
        
        # Process videos
        logger.info("\nStep 2: Processing videos...")
        self.process_videos(video_dir)
        
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
    parser.add_argument('--mode', choices=['images', 'videos', 'webcam', 'pipeline', 'all'], 
                        default='pipeline', help='Processing mode (default: pipeline)')
    parser.add_argument('--input', help='Input directory or file path')
    parser.add_argument('--conf', type=float, default=0.2, help='Confidence threshold')
    parser.add_argument('--cam', type=int, default=0, help='Camera index for webcam')
    parser.add_argument('--duration', type=int, default=30, help='Webcam capture duration (sec)')
    
    args = parser.parse_args()
    
    analytics = PPEDetectionAnalytics(conf_threshold=args.conf)
    
    if args.mode == 'images':
        analytics.process_images(args.input)
        analytics.save_report()
        analytics.save_text_report()
        analytics.print_summary()
    elif args.mode == 'videos':
        analytics.process_videos(args.input)
        analytics.save_report()
        analytics.save_text_report()
        analytics.print_summary()
    elif args.mode == 'webcam':
        analytics.process_webcam(camera_index=args.cam, duration_seconds=args.duration)
        analytics.save_report()
        analytics.save_text_report()
        analytics.print_summary()
    elif args.mode == 'all':
        analytics.process_images(args.input)
        analytics.process_videos(args.input)
        analytics.process_webcam(camera_index=args.cam, duration_seconds=args.duration)
        analytics.save_report()
        analytics.save_text_report()
        analytics.print_summary()
    else: # pipeline
        analytics.run_complete_pipeline(args.input)