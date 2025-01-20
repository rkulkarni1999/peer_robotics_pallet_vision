#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy

class DisplayNode(Node):
    def __init__(self):
        super().__init__('detection_display_node')

        self.declare_parameter('input_topic', '/detection_inference/overlay_image')

        # Get parameter
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        self.image_sub = self.create_subscription(Image, self.input_topic, self.image_callback, qos_profile)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Extract original dimensions
            original_height, original_width = cv_image.shape[:2]

            # Compute aspect ratio
            aspect_ratio = original_width / original_height

            # Define new width or height while maintaining aspect ratio
            target_width = 480
            target_height = int(target_width / aspect_ratio)
            
            # Resize the image
            resized_image = cv2.resize(cv_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Display the image
            cv2.imshow('Detection Results', resized_image)

            # Wait for a short period to process key events
            key = cv2.waitKey(1)

            # Optional: Add a key press to exit (e.g., ESC key)
            if key == 27: 
                self.get_logger().info('Exiting the display node.')
                rclpy.shutdown()
                cv2.destroyAllWindows()

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()

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