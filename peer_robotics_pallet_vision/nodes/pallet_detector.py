#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class DetectionNode(Node):
    def __init__(self):
        super().__init__('pallet_detector')

        # Parameters
        self.declare_parameter('rgb_topic', '/d455_1_rgb_image')
        self.declare_parameter('output_topic', '/detection_inference/overlay_image')

        # Get parameters
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # YOLO Model
        self.model = YOLO("yolo/models/final/pallet_detector_1_200.pt")
        self.bridge = CvBridge()

        # Subscriber
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)

        # Publisher
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)

    def rgb_callback(self, msg):
        # Convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Perform detection
        results = self.model.predict(source=cv_image, conf=0.5, save=False)
        annotated_image = self.process_results(cv_image, results)

        # Publish the annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.image_pub.publish(annotated_msg)

    def process_results(self, image, results):
        # Draw bounding boxes and labels on the image
        for result in results[0].boxes.data:
            box = result[:4].cpu().numpy().astype(int)  # Bounding box coordinates (x1, y1, x2, y2)
            score = result[4].cpu().item()             # Confidence score
            cls_id = int(result[5].cpu().item())       # Class ID

            # Define label and color
            label = f"{self.model.names[cls_id]}: {score:.2f}"
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Class-based color

            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw label
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
