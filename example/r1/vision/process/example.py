# Taken from https://support.unitree.com/home/en/R1_developer/video_service

#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time
import subprocess
import re
from typing import Tuple, Optional

class V4L2DepthReader:
    """Reads 16-bit depth maps from V4L2 loopback device (optimized version)"""
    
    def __init__(self, device_path: str = "/dev/video-dep"):
        self.device_path = device_path
        self.fd = None
        # Initialized with default values, will be auto-detected later
        self.width = 544
        self.height = 448
        self.depth = 2  # 16-bit depth, 2 bytes per pixel
        self.frame_size = self.width * self.height * self.depth
        self.last_frame = None  # Cache the last frame to avoid screen flickering

    def _get_resolution_from_v4l2ctl(self) -> Tuple[int, int]:
        """
        Execute v4l2-ctl command to parse the actual resolution of the device
        Returns: (width, height), returns default (544, 448) on failure
        """
        try:
            # Execute v4l2-ctl command to get format information
            result = subprocess.check_output(
                ['v4l2-ctl', '-d', self.device_path, '--list-formats-ext'],
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Regex to match resolution in format "Size: Discrete 640x352"
            pattern = r'Size:\s+Discrete\s+(\d+)x(\d+)'
            matches = re.findall(pattern, result)
            
            if matches:
                # Take the first matched resolution
                width = int(matches[0][0])
                height = int(matches[0][1])
                print(f"✅ Resolution obtained from v4l2-ctl: {width}x{height}")
                return width, height
            else:
                print("⚠️ No resolution info found, using default value: 544x448")
                return 544, 448
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to execute v4l2-ctl: {e.output}")
            print("⚠️ Using default resolution: 544x448")
            return 544, 448
        except FileNotFoundError:
            print("❌ v4l2-ctl tool not found, please install v4l-utils first")
            print("⚠️ Using default resolution: 544x448")
            return 544, 448
        except Exception as e:
            print(f"❌ Error parsing resolution: {str(e)}")
            print("⚠️ Using default resolution: 544x448")
            return 544, 448
        
    def open_device(self) -> bool:
        """Open the video device (non-blocking mode with auto resolution detection)"""
        # Step 1: Automatically get resolution
        self.width, self.height = self._get_resolution_from_v4l2ctl()
        self.frame_size = self.width * self.height * self.depth
        
        try:
            # Open device: O_RDWR=read-write, O_NONBLOCK=non-blocking
            self.fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
            print(f"✅ Device opened successfully: {self.device_path}")
            return True
        except OSError as e:
            print(f"❌ Failed to open device: {e}")
            # Try read-only mode
            try:
                self.fd = os.open(self.device_path, os.O_RDONLY | os.O_NONBLOCK)
                print(f"✅ Device opened in read-only mode: {self.device_path}")
                return True
            except OSError as e2:
                print(f"❌ Read-only mode also failed: {e2}")
                return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read frame data (with retry and timeout mechanism)
        Resolves [Errno 11] Resource temporarily unavailable issue
        """
        if not self.fd:
            print("Device not opened")
            return self.last_frame
        
        max_retries = 5  # Maximum number of retries
        retry_delay = 0.001  # Retry delay (seconds)
        read_timeout = 0.1  # Read timeout duration
        
        start_time = time.time()
        
        for retry in range(max_retries):
            try:
                # Attempt to read data
                data = os.read(self.fd, self.frame_size)
                
                # Data read successfully
                if len(data) == self.frame_size:
                    # Convert to 16-bit depth map
                    depth_frame = np.frombuffer(data, dtype=np.uint16)
                    depth_frame = depth_frame.reshape((self.height, self.width))
                    self.last_frame = depth_frame  # Update cache
                    return depth_frame
                
                # Incomplete data, partial data received
                elif len(data) > 0:
                    print(f"⚠️ Incomplete data: expected {self.frame_size} bytes, got {len(data)} bytes")
                    time.sleep(retry_delay)
                    continue
                
                # No data read
                else:
                    time.sleep(retry_delay)
                    continue
                    
            except BlockingIOError:
                # Resource temporarily unavailable, retry after short wait
                time.sleep(retry_delay)
                continue
                
            except Exception as e:
                print(f"⚠️ Frame read exception (retry {retry+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                continue
            
            # Check timeout
            if time.time() - start_time > read_timeout:
                break
        
        # All retries failed, return the last cached frame
        return self.last_frame
    
    def apply_colormap(self, depth_frame: np.ndarray, 
                       normalize_range: Tuple[int, int] = None) -> np.ndarray:
        """Apply pseudo-color mapping to the depth map"""
        if depth_frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Convert to float32 for processing
        depth_data = depth_frame.astype(np.float32)
        
        # Mark invalid depth values (0 and maximum value)
        invalid_mask = (depth_data == 0) | (depth_data == np.iinfo(np.uint16).max)
        
        # Calculate valid data range
        valid_data = depth_data[~invalid_mask]
        if len(valid_data) == 0:
            color_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            color_frame[invalid_mask] = [20, 20, 20]
            return color_frame
        
        # Determine normalization range
        if normalize_range is None:
            min_val = np.percentile(valid_data, 1)
            max_val = np.percentile(valid_data, 99)
        else:
            min_val, max_val = normalize_range
        
        # Normalize (avoid division by zero)
        depth_normalized = np.zeros_like(depth_data, dtype=np.float32)
        if max_val > min_val:
            depth_normalized[~invalid_mask] = np.clip(
                (depth_data[~invalid_mask] - min_val) / (max_val - min_val),
                0.0, 1.0
            )
        
        # Convert to 8-bit image
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        
        # Apply color mapping
        color_frame = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        # Set invalid areas to dark gray
        color_frame[invalid_mask] = [20, 20, 20]
        
        return color_frame
    
    def apply_colormap_custom(self, depth_frame: np.ndarray, 
                             normalize_range: Tuple[int, int] = None,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Custom color mapping"""
        if depth_frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        depth_data = depth_frame.astype(np.float32)
        invalid_mask = (depth_data == 0) | (depth_data == np.iinfo(np.uint16).max)
        valid_data = depth_data[~invalid_mask]
        
        if len(valid_data) == 0:
            color_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            color_frame[invalid_mask] = [20, 20, 20]
            return color_frame
        
        if normalize_range is None:
            min_val = np.percentile(valid_data, 1)
            max_val = np.percentile(valid_data, 99)
        else:
            min_val, max_val = normalize_range
        
        depth_normalized = np.zeros_like(depth_data, dtype=np.float32)
        if max_val > min_val:
            depth_normalized[~invalid_mask] = np.clip(
                (depth_data[~invalid_mask] - min_val) / (max_val - min_val),
                0.0, 1.0
            )
        
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        color_frame = cv2.applyColorMap(depth_8bit, colormap)
        color_frame[invalid_mask] = [20, 20, 20]
        
        return color_frame
    
    def draw_grid_with_depth(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                            num_horizontal: int = 5, num_vertical: int = 5) -> np.ndarray:
        """
        Draw grid lines and intersection depth values on the colorized depth map 
        (values displayed at top-right of intersections, divided by 1000)
        :param color_frame: Colorized depth map
        :param depth_frame: Raw depth data
        :param num_horizontal: Number of horizontal lines
        :param num_vertical: Number of vertical lines
        :return: Color image with grid drawn
        """
        if depth_frame is None:
            return color_frame
        
        # Create a copy to avoid modifying the original image
        frame_with_grid = color_frame.copy()
        h, w = depth_frame.shape
        
        # Calculate grid line positions (evenly distributed)
        # Horizontal lines (along y-axis)
        horizontal_steps = np.linspace(0, h-1, num_horizontal + 1, dtype=int)[1:-1]
        # Vertical lines (along x-axis)
        vertical_steps = np.linspace(0, w-1, num_vertical + 1, dtype=int)[1:-1]
        
        # Draw horizontal grid lines (white, semi-transparent)
        line_color = (255, 255, 255)  # White
        line_thickness = 1
        alpha = 0.7  # Line transparency
        
        # Create line layer
        line_layer = frame_with_grid.copy()
        
        # Draw horizontal lines
        for y in horizontal_steps:
            cv2.line(line_layer, (0, y), (w-1, y), line_color, line_thickness)
        
        # Draw vertical lines
        for x in vertical_steps:
            cv2.line(line_layer, (x, 0), (x, h-1), line_color, line_thickness)
        
        # Merge line layer and original image (semi-transparent effect)
        cv2.addWeighted(line_layer, alpha, frame_with_grid, 1 - alpha, 0, frame_with_grid)
        
        # Draw intersections and depth values (displayed at top-right, divided by 1000)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        offset_x = 8  # X offset of text relative to intersection (rightward)
        offset_y = -8  # Y offset of text relative to intersection (upward)
        depth_scale = 1000.0  # Depth value scaling factor (divided by 1000)
        
        # Iterate over all intersections
        for y in horizontal_steps:
            for x in vertical_steps:
                # Get depth value at this point
                depth_value = depth_frame[y, x]
                
                # Check if depth value is valid
                if depth_value == 0 or depth_value == np.iinfo(np.uint16).max:
                    depth_text = "---"
                else:
                    # Divide depth value by 1000, display with 3 decimal places
                    depth_scaled = depth_value / depth_scale
                    depth_text = f"{depth_scaled:.3f}"
                
                # Get text size
                text_size = cv2.getTextSize(depth_text, font, font_scale, font_thickness)[0]
                
                # Calculate text position (top-right of intersection)
                text_x = x + offset_x
                text_y = y + offset_y
                
                # Adjust text baseline position (OpenCV y-coordinate is text baseline position)
                text_y = text_y + text_size[1]
                
                # Ensure text is within image bounds
                text_x = max(2, min(w - text_size[0] - 2, text_x))
                text_y = max(text_size[1] + 2, min(h - 2, text_y))
                
                # Draw text background (improve readability)
                bg_rect = (
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2)
                )
                cv2.rectangle(frame_with_grid, bg_rect[0], bg_rect[1], bg_color, -1)
                
                # Draw depth value text
                cv2.putText(frame_with_grid, depth_text, (text_x, text_y), 
                           font, font_scale, text_color, font_thickness)
                
                # Draw intersection marker (small circle)
                cv2.circle(frame_with_grid, (x, y), 3, (0, 255, 0), -1)  # Green filled circle
        
        return frame_with_grid
    
    def close(self):
        """Close the device"""
        if self.fd:
            os.close(self.fd)
            self.fd = None
            print("Device closed")

def main():
    """Main function"""
    # Create depth reader
    depth_reader = V4L2DepthReader("/dev/video-dep")
    
    # Open device
    if not depth_reader.open_device():
        return
    
    # Create window (adaptive resolution)
    cv2.namedWindow("Depth View (Colorized)", cv2.WINDOW_NORMAL)
    # Adjust window size based on detected resolution (1.5x scaling)
    win_width = int(depth_reader.width * 1.5)
    win_height = int(depth_reader.height * 1.5)
    cv2.resizeWindow("Depth View (Colorized)", win_width, win_height)
    
    # Configuration
    NORMALIZE_RANGE = None  # Auto range, can be manually set e.g. (0, 2000)
    
    # Color mapping configurations
    colormap_configs = [
        (cv2.COLORMAP_JET, "JET"),
        (cv2.COLORMAP_VIRIDIS, "VIRIDIS"),
        (cv2.COLORMAP_HOT, "HOT"),
        (cv2.COLORMAP_RAINBOW, "RAINBOW"),
        (cv2.COLORMAP_MAGMA, "MAGMA"),
        (cv2.COLORMAP_INFERNO, "INFERNO")
    ]
    current_colormap_idx = 0
    
    print("="*50)
    print("Starting depth stream reading...")
    print(f"Current device resolution: {depth_reader.width}x{depth_reader.height}")
    print("Key instructions:")
    print("  q - Quit program")
    print("  s - Save current frame")
    print("  c - Switch color mapping mode")
    print("  +/- - Adjust normalization range")
    print("  Depth values displayed in meters (divided by 1000)")
    print("="*50)
    
    save_count = 0
    
    try:
        while True:
            # Read depth frame
            depth_frame = depth_reader.read_frame()
            
            # Generate color image
            if depth_frame is not None:
                # Get current color mapping
                colormap, colormap_name = colormap_configs[current_colormap_idx]
                
                # Apply color mapping
                color_frame = depth_reader.apply_colormap_custom(
                    depth_frame, NORMALIZE_RANGE, colormap
                )
                
                # Dynamically adjust grid count based on resolution (avoid too dense/sparse grid)
                grid_h = max(3, min(8, depth_reader.height // 100))
                grid_v = max(3, min(8, depth_reader.width // 100))
                
                # Draw grid and depth values
                color_frame_with_grid = depth_reader.draw_grid_with_depth(
                    color_frame, depth_frame, num_horizontal=grid_h, num_vertical=grid_v
                )
                
                # Add resolution and colormap info
                info_text = f"Res: {depth_reader.width}x{depth_reader.height} | Colormap: {colormap_name}"
                cv2.putText(color_frame_with_grid, info_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display depth map with grid
                cv2.imshow("Depth View (Colorized)", color_frame_with_grid)
            else:
                # No valid frame, show black screen with prompt
                black_frame = np.zeros((depth_reader.height, depth_reader.width, 3), dtype=np.uint8)
                cv2.putText(black_frame, "NO FRAME DATA", (50, depth_reader.height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Depth View (Colorized)", black_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                if depth_frame is not None:
                    # Save color image with grid
                    color_frame = depth_reader.apply_colormap_custom(
                        depth_frame, NORMALIZE_RANGE, colormap_configs[current_colormap_idx][0]
                    )
                    color_frame_with_grid = depth_reader.draw_grid_with_depth(
                        color_frame, depth_frame, num_horizontal=5, num_vertical=5
                    )
                    # Filename includes resolution info
                    cv2.imwrite(f"depth_color_{save_count:04d}_{depth_reader.width}x{depth_reader.height}.png", color_frame_with_grid)
                    # Save raw depth data
                    np.save(f"depth_raw_{save_count:04d}_{depth_reader.width}x{depth_reader.height}.npy", depth_frame)
                    print(f"Frame {save_count:04d} saved (resolution: {depth_reader.width}x{depth_reader.height})")
                    save_count += 1
            elif key == ord('c'):
                # Switch color mapping
                current_colormap_idx = (current_colormap_idx + 1) % len(colormap_configs)
                colormap, colormap_name = colormap_configs[current_colormap_idx]
                print(f"Switched to color mapping: {colormap_name}")
            elif key == ord('+') or key == ord('='):
                # Expand normalization range
                if NORMALIZE_RANGE is not None:
                    NORMALIZE_RANGE = (NORMALIZE_RANGE[0], NORMALIZE_RANGE[1] * 1.2)
                    print(f"Normalization range: {NORMALIZE_RANGE}")
            elif key == ord('-'):
                # Shrink normalization range
                if NORMALIZE_RANGE is not None:
                    NORMALIZE_RANGE = (NORMALIZE_RANGE[0], NORMALIZE_RANGE[1] * 0.8)
                    print(f"Normalization range: {NORMALIZE_RANGE}")
                    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        depth_reader.close()
        cv2.destroyAllWindows()
        print("Program exited")

if __name__ == "__main__":
    # Check if v4l2-ctl is installed (optional)
    try:
        # subprocess.check_call(['v4l2-ctl', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pass
    except FileNotFoundError:
        print("⚠️ Warning: v4l2-ctl tool not detected, recommended to install: sudo apt install v4l-utils")
        print("⚠️ Default resolution 544x448 will be used\n")
    
    main()