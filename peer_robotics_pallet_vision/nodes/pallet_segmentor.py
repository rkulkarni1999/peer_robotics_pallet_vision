#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('pallet_segmentor')
    
        # Parameters
        self.declare_parameter('rgb_topic', '/d455_1_rgb_image')
        self.declare_parameter('depth_topic', '/d455_1_depth_image')
        self.declare_parameter('output_topic', '/segmentation_inference/overlay_image')
        
         # Get parameters
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # YOLO Model
        self.model = YOLO("yolo/models/final/pallet_segmentation.pt")
        self.bridge = CvBridge()
        
        # Subscribers
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        
        # publisher
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)
        
        # latest depth data
        self.latest_depth = None
        
        self.counter = 0
        
    def depth_callback(self, msg):
        
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    def rgb_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        results = self.model.predict(source=cv_image, conf=0.4, save=False)
        annotated_image = self.process_results(cv_image, results)
     
        # Publish the annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.image_pub.publish(annotated_msg)
        
        
    def process_results(self, image, results):
        # Initialize a semantic mask for each class
        semantic_masks = {0: np.zeros(image.shape[:2], dtype=np.uint8),  # Ground
                        1: np.zeros(image.shape[:2], dtype=np.uint8)}  # Pallet

        # Combine masks for the same class
        for mask, cls_id in zip(results[0].masks.data, results[0].boxes.cls):
            cls_id = int(cls_id)
            mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert to binary mask (0 or 1)
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            semantic_masks[cls_id] = cv2.bitwise_or(semantic_masks[cls_id], mask_resized)

        # Overlay each class mask and label the entire region
        for cls_id, semantic_mask in semantic_masks.items():
            if np.any(semantic_mask):  # Check if the mask contains any pixels
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Colors: Ground (green), Pallet (blue)
                label = "Ground" if cls_id == 0 else "Pallet"

                # Create a colored overlay for the mask
                overlay = np.zeros_like(image, dtype=np.uint8)
                overlay[semantic_mask == 1] = color

                # Blend the overlay with the original image
                image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

                # Calculate centroid for labeling
                moments = cv2.moments(semantic_mask.astype(np.uint8))
                if moments["m00"] > 0:  # Ensure the area is not zero
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])
                    cv2.putText(
                        image,
                        label,
                        (centroid_x, centroid_y),  # (x, y) format for OpenCV
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
        return image



def main(args=None):
    
    rclpy.init(args=args)
    node = SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
        