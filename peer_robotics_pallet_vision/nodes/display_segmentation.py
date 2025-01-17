#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DisplayNode(Node):
    def __init__(self):
        super().__init__('segmentation_display_node')

        # Parameters
        self.declare_parameter('input_topic', '/segmentation_inference/overlay_image')

        # Get parameter
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        # CVBridge for image conversion
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = self.create_subscription(Image, self.input_topic, self.image_callback, 10)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Display the image
            cv2.imshow('Detection Results', cv_image)

            # Wait for a short period to process key events
            key = cv2.waitKey(1)

            # Optional: Add a key press to exit (e.g., ESC key)
            if key == 27:  # ESC key
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