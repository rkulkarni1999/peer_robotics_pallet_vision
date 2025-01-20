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

from rclpy.qos import QoSProfile, ReliabilityPolicy


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('pallet_segmentor')
    
        self.declare_parameter('rgb_topic', '/robot1/zed2i/left/image_rect_color')
        self.declare_parameter('depth_topic', '/d455_1_depth_image')
        self.declare_parameter('output_topic', '/segmentation_inference/overlay_image')
        
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        self.model = YOLO("yolo/models/final/segmentation/segmentation_final.pt")
        # self.model = YOLO("yolo/models/final/segmentation/segmentation_final.pt")

        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  
            depth=10
        )
        
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile)
        
        self.image_pub = self.create_publisher(Image, self.output_topic, qos_profile)
        
        self.latest_depth = None
        
        self.counter = 0
        
    def depth_callback(self, msg):
        
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    def rgb_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        results = self.model.predict(source=cv_image, conf=0.60, save=False)
        annotated_image = self.process_results(cv_image, results)
     
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.image_pub.publish(annotated_msg)
        
        
    def process_results(self, image, results):
        
        semantic_masks = {0: np.zeros(image.shape[:2], dtype=np.uint8),  # Ground
                        1: np.zeros(image.shape[:2], dtype=np.uint8)}  # Pallet

        for mask, cls_id in zip(results[0].masks.data, results[0].boxes.cls):
            cls_id = int(cls_id)
            mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert to binary mask (0 or 1)
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            semantic_masks[cls_id] = cv2.bitwise_or(semantic_masks[cls_id], mask_resized)

        for cls_id, semantic_mask in semantic_masks.items():
            if np.any(semantic_mask):  
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Colors: Ground (green), Pallet (blue)
                label = "Ground" if cls_id == 0 else "Pallet"

                overlay = np.zeros_like(image, dtype=np.uint8)
                overlay[semantic_mask == 1] = color

                image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

                moments = cv2.moments(semantic_mask.astype(np.uint8))
                if moments["m00"] > 0:  # Ensure the area is not zero
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])
                    cv2.putText(
                        image,
                        label,
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  
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
    
        