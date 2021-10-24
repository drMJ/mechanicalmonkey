from mmky import writers
import cv2

if __name__ == '__main__':
    episode = writers.readRobosuite('d:\\code\\mechanicalmonkey\\mmky\\trajectories\\robosuite_pour_1_1635058836.5488708.hdf5')
    for image in episode['images']:
        cv2.imshow("image", image)
        cv2.waitKey(16)

    # episode = writers.readSimpleNpy('C:\\code\\mechanicalmonkey\\mmky\\trajectories\\cup_pour_simple.npy')
    # for image in episode['images']:
    #     cv2.imshow("image", image)
    #     #print(episode["actions"])
    #     cv2.waitKey(16)
