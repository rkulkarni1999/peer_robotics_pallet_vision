#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy

class DetectionNode(Node):
    def __init__(self):
        super().__init__('pallet_detector')

        # Parameters
        self.declare_parameter('rgb_topic', '/robot1/zed2i/left/image_rect_color')
        self.declare_parameter('depth_topic', '/d455_1_depth_image')
        self.declare_parameter('output_topic', '/detection_inference/overlay_image')

        # Get parameters
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # YOLO Model
        self.model = YOLO("yolo/models/final/detection/detection_final.pt")
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Use BEST_EFFORT to match most sensor topics
            depth=10
        )

        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile)
        
        self.image_pub = self.create_publisher(Image, self.output_topic, qos_profile)
        
        self.latest_depth = None

    def depth_callback(self, msg):
        
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    def rgb_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        results = self.model.predict(source=cv_image, conf=0.55, save=False)
        annotated_image = self.process_results(cv_image, results)

        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.image_pub.publish(annotated_msg)

    def process_results(self, image, results):

        for result in results[0].boxes.data:
            box = result[:4].cpu().numpy().astype(int)  
            score = result[4].cpu().item()             
            cls_id = int(result[5].cpu().item())       

            label = f"{self.model.names[cls_id]}: {score:.2f}"
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

            cv2.putText(
                image,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA
            )

        return image


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()

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
