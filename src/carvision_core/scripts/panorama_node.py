#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer


class MoraiPanoramaNode:
    def __init__(self):
        # 파라미터에서 토픽 이름 받아오기 (rosparam 없으면 기본값 사용)
        left_topic  = rospy.get_param("~left_image_topic",  "/morai/camera/left/image_raw")
        right_topic = rospy.get_param("~right_image_topic", "/morai/camera/right/image_raw")

        rospy.loginfo("Left  image topic : %s", left_topic)
        rospy.loginfo("Right image topic : %s", right_topic)

        self.bridge = CvBridge()

        # SIFT 생성 (contrib 필요)
        try:
            self.descriptor = cv2.xfeatures2d.SIFT_create()
            rospy.loginfo("Using SIFT feature detector.")
        except AttributeError:
            rospy.logwarn("SIFT not available. Falling back to ORB.")
            self.descriptor = cv2.ORB_create(2000)

        # 매칭기 생성
        self.matcher = cv2.DescriptorMatcher_create("BruteForce")

        # 이미지 동기화 구독자 설정
        sub_left  = Subscriber(left_topic, Image)
        sub_right = Subscriber(right_topic, Image)

        # 타임스탬프가 완전히 같지 않아도 근처 프레임을 같이 묶어주는 ApproximateTimeSynchronizer 사용
        self.ts = ApproximateTimeSynchronizer(
            [sub_left, sub_right],
            queue_size=10,
            slop=0.05  # 초 단위 허용 오차
        )
        self.ts.registerCallback(self.callback)

        # OpenCV 윈도우 생성
        cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)

    def stitch(self, imgL, imgR):
        """왼쪽/오른쪽 이미지를 받아 파노라마 이미지로 합침."""
        hl, wl = imgL.shape[:2]
        hr, wr = imgR.shape[:2]

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # 특징점 및 디스크립터 추출
        kpsL, featuresL = self.descriptor.detectAndCompute(grayL, None)
        kpsR, featuresR = self.descriptor.detectAndCompute(grayR, None)

        if featuresL is None or featuresR is None:
            rospy.logwarn("No features found in one of the images.")
            return imgL

        # knn 매칭
        matches = self.matcher.knnMatch(featuresR, featuresL, 2)

        # 좋은 매칭 선별 (Lowe ratio test)
        good_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good_matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(good_matches) > 4:
            ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches])
            ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches])

            # 호모그래피 계산
            mtrx, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

            if mtrx is None:
                rospy.logwarn("Homography matrix is None. Returning left image only.")
                return imgL

            # 오른쪽 이미지를 왼쪽 좌표계로 warp
            pano_width = wl + wr
            pano_height = max(hl, hr)
            panorama = cv2.warpPerspective(imgR, mtrx, (pano_width, pano_height))

            # 왼쪽 이미지 덮어쓰기
            panorama[0:hl, 0:wl] = imgL

            return panorama
        else:
            rospy.logwarn("Not enough good matches (%d). Returning left image only.", len(good_matches))
            return imgL

    def callback(self, left_msg, right_msg):
        """동기화된 왼쪽/오른쪽 이미지 콜백."""
        try:
            imgL = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
            imgR = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))
            return

        # 파노라마 생성
        pano = self.stitch(imgL, imgR)

        # OpenCV로 뿌리기
        cv2.imshow("Left", imgL)
        cv2.imshow("Right", imgR)
        cv2.imshow("Panorama", pano)
        cv2.waitKey(1)  # 실시간 갱신


def main():
    rospy.init_node("morai_panorama_node", anonymous=False)
    rospy.loginfo("MORAI Panorama Node started.")
    node = MoraiPanoramaNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down MORAI Panorama Node.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
