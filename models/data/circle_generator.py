import numpy as np
from matplotlib import pyplot as plt


def gen_circles(number, size=100, radius=None, location=None):
    frame = np.zeros([size, size, 1])
    centricity = 0.85

    def create_mask(center):
        Y, X = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= rad
        return mask

    i = 0
    while i < number:
        if location is None:
            center = np.random.randint(size*centricity, size=2)+(1-centricity)*size
        else:
            center = np.array([l for l in location])
        if radius is None:
            rad = np.random.randint(size/3)+(size/10)
        else:
            rad = radius

        mask = create_mask(center)
        circle = frame.copy()
        circle[mask] = 1
        #yield i, circle
        yield circle
        i += 1



if __name__ == '__main__':

    size = 60
    cols = 4
    rows = 3
    fig = plt.figure(figsize=(10, 10))

    #for index, circle in gen_circles(cols*rows, size):
    index = 0
    for circle in gen_circles(cols*rows, size, radius=15):
        print("img added")
        fig.add_subplot(rows, cols, index+1)
        plt.imshow(circle.reshape([size, size]))
        index += 1
    plt.show()
